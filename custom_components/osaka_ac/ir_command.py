import io
import base64
from bisect import bisect
from struct import pack, unpack
from enum import Enum


headers = [3294, 1708]


def strip_headers(signal: list[int]) -> list[int]:
    if len(signal) != 227:
        raise ValueError(f'signal has wrong length, expected 227, got {len(signal)}')
    if len(signal) == 225:
        raise ValueError(f'Headers seems to be already mising')
    return signal[2:]


def add_headers(signal: list[int]) -> list[int]:
    if len(signal) != 225:
        raise ValueError(f'signal has wrong length, expected 225, got {len(signal)}')
    if len(signal) == 227:
        raise ValueError(f'Headers seem to be already included')
    signal.insert(0, headers[0])
    signal.insert(1, headers[1])
    return signal


def convert_to_binary(signal: list[int]) -> list[int]:
    # Convert values 350-550 to 0 and 1150-1350 to 1
    result = [0 if x in range(350, 550) else 1 if x in range(1150, 1350) else x for x in signal]
    return result


def convert_to_intervals(signal: list[int]) -> list[int]:
    # Revert conversion to binary, replacing 0 with 467 and 1 with 1261
    result = [467 if x==0 else 1261 if x==1 else x for x in signal]
    return result


def remove_spacings(signal: list[int]) -> list[int]:
    if len(signal) != 225:
        raise ValueError(f'signal has wrong length, expected 225, got {len(signal)}')
    if signal[0] != 0:
        raise ValueError(f'Signal at index 0 should be 0, check the signal')
    return signal[1::2]


def restore_spacings(signal: list[int]) -> list[int]:
    if len(signal) != 112:
        raise ValueError(f'signal has wrong length, expected 112 got {len(signal)}')
    result = [x for x in signal]
    for i in range(0,len(result)*2,2):
        result.insert(i, 0)
    result.append(0)
    return result


def compute_checksum(binary_str: str) -> str:
    """
    Given a binary string (e.g., "11000100...") where the length is a multiple of 8,
    computes the checksum as the sum of all preceding bytes modulo 256.
    Assumes that binary_str does not include the checksum (final 8 bits).
    Returns an 8-bit binary string (e.g., "10001010").
    """
    if len(binary_str) != 104-8 and len(binary_str) % 8 != 0:
        raise ValueError(f'binary_str has wrong length, expected 104 got {len(binary_str)}')
    total = 0
    binary_str = binary_str[::-1]
    # Process every 8-bit block:
    for i in range(0, len(binary_str), 8):
        byte_val = int(binary_str[i:i+8], 2)
        total += byte_val
    checksum = total % 256
    return f"{checksum:08b}"[::-1]


def strip_checksum(signal: list[int]) -> list[int]:
    if len(signal) != 112:
        raise ValueError(f'signal has wrong length, expected 112 got {len(signal)}')
    return signal[:-8]

def decode_ir(code: str) -> list[int]:
    '''
    Decodes an IR code string from a Tuya blaster.
    Returns the IR signal as a list of µs durations,
    with the first duration belonging to a high state.
    '''
    payload = base64.decodebytes(code.encode('ascii'))
    payload = decompress(io.BytesIO(payload))

    signal = []
    while payload:
        assert len(payload) >= 2, \
            f'garbage in decompressed payload: {payload.hex()}'
        signal.append(unpack('<H', payload[:2])[0])
        payload = payload[2:]
    return signal

def encode_ir(signal: list[int], compression_level=2) -> str:
    '''
    Encodes an IR signal (see `decode_tuya_ir`)
    into an IR code string for a Tuya blaster.
    '''
    payload = b''.join(pack('<H', t) for t in signal)
    compress(out := io.BytesIO(), payload, compression_level)
    payload = out.getvalue()
    return base64.encodebytes(payload).decode('ascii').replace('\n', '')

# DECOMPRESSION

def decompress(inf: io.FileIO) -> bytes:
    '''
    Reads a "Tuya stream" from a binary file,
    and returns the decompressed byte string.
    '''
    out = bytearray()

    while (header := inf.read(1)):
        L, D = header[0] >> 5, header[0] & 0b11111
        if not L:
            # literal block
            L = D + 1
            data = inf.read(L)
            assert len(data) == L
        else:
            # length-distance pair block
            if L == 7:
                L += inf.read(1)[0]
            L += 2
            D = (D << 8 | inf.read(1)[0]) + 1
            data = bytearray()
            while len(data) < L:
                data.extend(out[-D:][:L-len(data)])
        out.extend(data)

    return bytes(out)


# COMPRESSION

def emit_literal_blocks(out: io.FileIO, data: bytes):
    for i in range(0, len(data), 32):
        emit_literal_block(out, data[i:i+32])

def emit_literal_block(out: io.FileIO, data: bytes):
    length = len(data) - 1
    assert 0 <= length < (1 << 5)
    out.write(bytes([length]))
    out.write(data)

def emit_distance_block(out: io.FileIO, length: int, distance: int):
    distance -= 1
    assert 0 <= distance < (1 << 13)
    length -= 2
    assert length > 0
    block = bytearray()
    if length >= 7:
        assert length - 7 < (1 << 8)
        block.append(length - 7)
        length = 7
    block.insert(0, length << 5 | distance >> 8)
    block.append(distance & 0xFF)
    out.write(block)

def compress(out: io.FileIO, data: bytes, level=2):
    '''
    Takes a byte string and outputs a compressed "Tuya stream".
    Implemented compression levels:
    0 - copy over (no compression, 3.1% overhead)
    1 - eagerly use first length-distance pair found (linear)
    2 - eagerly use best length-distance pair found
    3 - optimal compression (n^3)
    '''
    if level == 0:
        return emit_literal_blocks(out, data)

    W = 2**13 # window size
    L = 255+9 # maximum length
    distance_candidates = lambda: range(1, min(pos, W) + 1)

    def find_length_for_distance(start: int) -> int:
        length = 0
        limit = min(L, len(data) - pos)
        while length < limit and data[pos + length] == data[start + length]:
            length += 1
        return length
    find_length_candidates = lambda: \
        ( (find_length_for_distance(pos - d), d) for d in distance_candidates() )
    find_length_cheap = lambda: \
        next((c for c in find_length_candidates() if c[0] >= 3), None)
    find_length_max = lambda: \
        max(find_length_candidates(), key=lambda c: (c[0], -c[1]), default=None)

    if level >= 2:
        suffixes = []; next_pos = 0
        key = lambda n: data[n:]
        find_idx = lambda n: bisect(suffixes, key(n), key=key)
        def distance_candidates():
            nonlocal next_pos
            while next_pos <= pos:
                if len(suffixes) == W:
                    suffixes.pop(find_idx(next_pos - W))
                suffixes.insert(idx := find_idx(next_pos), next_pos)
                next_pos += 1
            idxs = (idx+i for i in (+1,-1)) # try +1 first
            return (pos - suffixes[i] for i in idxs if 0 <= i < len(suffixes))

    if level <= 2:
        find_length = { 1: find_length_cheap, 2: find_length_max }[level]
        block_start = pos = 0
        while pos < len(data):
            if (c := find_length()) and c[0] >= 3:
                emit_literal_blocks(out, data[block_start:pos])
                emit_distance_block(out, c[0], c[1])
                pos += c[0]
                block_start = pos
            else:
                pos += 1
        emit_literal_blocks(out, data[block_start:pos])
        return

    # use topological sort to find shortest path
    predecessors = [(0, None, None)] + [None] * len(data)
    def put_edge(cost, length, distance):
        npos = pos + length
        cost += predecessors[pos][0]
        current = predecessors[npos]
        if not current or cost < current[0]:
            predecessors[npos] = cost, length, distance
    for pos in range(len(data)):
        if c := find_length_max():
            for l in range(3, c[0] + 1):
                put_edge(2 if l < 9 else 3, l, c[1])
        for l in range(1, min(32, len(data) - pos) + 1):
            put_edge(1 + l, l, 0)

    # reconstruct path, emit blocks
    blocks = []; pos = len(data)
    while pos > 0:
        _, length, distance = predecessors[pos]
        pos -= length
        blocks.append((pos, length, distance))
    for pos, length, distance in reversed(blocks):
        if not distance:
            emit_literal_block(out, data[pos:pos + length])
        else:
            emit_distance_block(out, length, distance)


def check_compare_checksum(signal_full):
    if len(signal_full) != 112:
        raise ValueError(f"Expected 112 bits long signal to check checksum, got {len(signal_full)}")
    checksum_len = 8
    checksum_original = signal_full[-checksum_len:]
    signal = signal_full[:-checksum_len]
    checksum = compute_checksum(signal)
    return checksum_original == checksum


def decode_full_process(raw_signal: str) -> list[int]:
    # Process raw signal into cleared, parsed list of bits
    try:
        signal = decode_ir(raw_signal)
    except Exception:
        raise ValueError(f'Corrupt signal {raw_signal}')
    signal = strip_headers(signal)
    signal = convert_to_binary(signal)
    signal = remove_spacings(signal)
    signal = strip_checksum(signal)
    return signal


def encode_full_process(clean_signal: list[int]) -> str:
    # Revert full decoding process, resulting in ready to use base64 command
    signal = clean_signal.copy()
    # Add checksum
    [signal.append(int(x)) for x in compute_checksum(''.join([str(b) for b in signal]))]
    signal = restore_spacings(signal)
    signal = convert_to_intervals(signal)
    signal = add_headers(signal)
    return encode_ir(signal)

# BITS POSITIONS (inverted, 111-x to get original) FOR HEAT MODE
# ================
# Basic acts as a template (when temp, fan mode and night mode only researched so far)
# So far bits seems to be static:
# 0-35:   0...0
# 36:     1
# 37-54:  0...0
# 55-61:  1001001
# 62-78:  0...0
# 79-103: 1001001101100101100100011
#         1001001101100101100100011
# POWER
# Index 61 and 98(?, no, its part of "hashsum" part of signal) for on/off diff, 1 for ON
# SET TEMP
# Index 44-47 (including) in binary temperature, where 0000 - 31 degrees, 1111 - 16 degrees
# FAN SPEED SET (HEAT MODE)
# Index 37-39 (including) in binary fan mode, auto - 000, low 010, med 011, high 101, step 2 bits
# SET NIGHT MODE
# Index 39 0 - off, 1 - on
# SET TURBO? MODE
# Index 12 - 1: on, 0: off
# Index 37-38 - 11: on, 00: off
# SET SWING MODE
# Index 34-36: auto-0-30-45-60-90-swing with step i starting from 000
#    horizontal: 001
#    30deg:      010
#    45deg:      011
#    60deg:      100
#    90deg:      101
#    swing:      111
#    auto:       000
# ================================
# COOL MODE DIFF
# Index 54-55 seems to be static '1'
# Index 48-53 static 0
# ================================
# FAN MODE DIFF
# Index 53-55 static 1
# Index for temp bits 44-47 locked to 101 (22)
# Only fan speed, night mode and swing settings are allowed to be changed
# ================================
# FEEL MODE DIFF
# Again, only fan speed, night mode and swing settings are available
# Index 48-55: 00001000, index 52 = 1
# Index 40-47: 00000111, index 40-44: 0s, index 45-47: 1s
# ================================
# DRY MODE DIFF
# Again, only fan speed, night mode and swing settings are available
# Index 40-47: 00001001, index 44,47 - 1, others (40-43, 45-46) - 0
# Index 48-55: 00000010: index 54: 1, 48-53: 0, 55: 0


def turn_off(raw):
    decoded = decode_full_process(raw)
    decoded[42] = 0
    decoded[106] = 0
    return encode_full_process(decoded)

def turn_on(raw):
    decoded = decode_full_process(raw)
    decoded[42] = 1
    decoded[106] = 1
    return encode_full_process(decoded)


def generate_ir_command(hvac_mode, target_temp, fan_mode, swing_mode, power_on=True, turbo=False, night_mode=False):
    """
    Generate a Base64-encoded IR command to control the AC.

    Parameters:
      - hvac_mode: string, e.g., "heat", "cool", "fan_only", "feel", "dry"
      - target_temp: int, target temperature in °C.
      - fan_mode: string, e.g., "auto", "low", "med", "high"
      - swing_mode: string, e.g., "horizontal", "30deg", "45deg", "60deg", "90deg", "swing", "auto"
      - power_on: bool, whether AC should be turned on.
      - turbo: bool, whether turbo mode is active.
      - night_mode: bool, whether night mode is active.

    Returns:
      A Base64-encoded string representing the IR command.
    """
    # Use a base template (for example, the uncompressed payload for HEAT mode with default settings).
    # This is an example binary string (as a Python string of 0s and 1s).
    # 0         1         2         3         4         5         6         7         8         9         0
    # 01234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123
    # 00000000000000000000000000000000000110110000100000000011001001000000000000000001001001101100101100100011
    base_template = (
        "0" * 34
        + "001"  # Swing mode - horizontal
        + "000"  # Fan speed mode - auto
        + "0000" # 40-44 bits | Prefix before temperature bits
        + "0100" # 44-47 bits | 27 degrees
        + "00000001" # 48-55 bits | Heat mode
        + "00100100" # 56-63 bits | static & power
        + "0000000000000001001001101100101100100011" # signal ending
    )
    fixed_length = 104
    if len(base_template) < fixed_length:
        raise ValueError(f"base_template too short: {len(base_template)}")

    if not power_on:
        signal = base_template[:61] + '0' + base_template[62:]
        return encode_full_process([int(x) for x in signal[::-1]])

    # SET HVAC MODE
    hvac_modes = {
        'feel': '00001000',
        'cool': '00000011',
        'dry': '00000010',
        'heat': '00000001',
        'fan_only': '00000111'
    }
    allowed_modes = hvac_modes.keys()

    if hvac_mode == 'auto':
        hvac_mode = 'feel'
    if hvac_mode not in allowed_modes:
        raise ValueError(f"hvac_mode must be one of {allowed_modes}, got {hvac_mode}")
    signal = base_template[:48] + hvac_modes[hvac_mode] + base_template[56:]

    if hvac_mode in ('heat', 'cool'):
        # SET TEMPERATURE
        # Assume that the temperature field covers bits 44-47 and is encoded as (temp - offset) in 4 bits.
        starting_temp = 31
        temp_value = starting_temp - target_temp
        new_temp_bits = format(temp_value, "04b")  # e.g., for 27°C: (31-27)=4 → "0100"
        # Replace bits 44-47 in the template:
        signal = signal[:44] + new_temp_bits + signal[48:]
    elif hvac_mode in ('fan_only', 'dry'):
        signal = signal[:40] + '00001001' + signal[48:]
    elif hvac_mode == 'feel':
        signal = signal[:40] + '00000111' + signal[48:]

    # SET swing mode
    swing_modes = {
        'horizontal': '001',
        '30deg':      '010',
        '45deg':      '011',
        '60deg':      '100',
        '90deg':      '101',
        'swing':      '111',
        'auto':       '000'
    }
    if swing_mode not in swing_modes:
        raise ValueError(f"swing_mode must be one of {swing_modes}, got {swing_mode}")
    signal = signal[:34] + swing_modes[swing_mode] + signal[37:]

    # SET fan apeed, turbo, night mode
    if night_mode and turbo:
        raise ValueError("Cannot set night mode and turbo at the same time")
    if night_mode:
        signal = signal[:37] + '001' + signal[40:]
    elif turbo:
        if hvac_mode in ('fan_only', 'feel', 'dry'):
            raise ValueError(f"Turbo mode is available only for cool and heat modes, got mode {hvac_mode}")
        signal = signal[:37] + '110' + signal[40:]
        signal = signal[:12] + '1' + signal[13:]
    else:
        fan_modes = {
            'auto': '000',
            'low': '010',
            'medium': '011',
            'high': '101'
        }
        if fan_mode not in fan_modes:
            raise ValueError(f"fan_mode must be one of {fan_modes}")
        signal = signal[:37] + fan_modes[fan_mode] + signal[40:]

    # Compute checksum for updated payload
    if len(signal) != fixed_length:
        raise ValueError(f"Received invalid signal length after signal creation. Signal must have {fixed_length} bits")
    signal = encode_full_process([int(x) for x in signal[::-1]])

    ## Convert final_payload (a string of bits) to bytes:
    #def bits_to_bytes(bits):
    #    return int(bits, 2).to_bytes(len(bits) // 8, byteorder="big")

    #payload_bytes = bits_to_bytes(final_payload)
    #import base64
    #return base64.b64encode(payload_bytes).decode("utf-8")
    return signal


def parse_encoded_command(command_raw: str) -> dict:
    """Parses encoded IR command into a dictionary of feature states."""

    # Decode and reverse signal to match bit order
    signal = decode_full_process(command_raw)
    signal = ''.join([str(x) for x in signal[::-1]])  # Reverse back to correct order

    command = {}

    # HVAC Modes Mapping
    hvac_modes = {
        '00001000': 'feel',
        '00000011': 'cool',
        '00000010': 'dry',
        '00000001': 'heat',
        '00000111': 'fan_only'
    }
    command['hvac_mode'] = hvac_modes.get(signal[48:56], None)

    # Temperature Calculation (bits 44-47, inverted formula)
    if command['hvac_mode'] in ('cool', 'heat'):
        temp_value = int(signal[44:48], 2)
        command['target_temp'] = 31 - temp_value  # Reverse the encoding formula
    else:
        command['target_temp'] = None  # No target temperature for fan_only or dry modes

    # Swing Modes Mapping (bits 34-36)
    swing_modes = {
        '001': 'horizontal',
        '010': '30deg',
        '011': '45deg',
        '100': '60deg',
        '101': '90deg',
        '111': 'swing',
        '000': 'auto'
    }
    command['swing_mode'] = swing_modes.get(signal[34:37], None)

    # Fan Modes Mapping (bits 37-39)
    fan_modes = {
        '000': 'auto',
        '010': 'low',
        '011': 'medium',
        '101': 'high'
    }
    command['fan_mode'] = fan_modes.get(signal[37:40], None)

    # Power State (bit 61)
    command['power_on'] = signal[61] == '1'

    # Turbo Mode (bit 12 - "1" means turbo is enabled)
    command['turbo'] = signal[12] == '1'

    # Night Mode (bit 37-39, "001" indicates night mode is active)
    command['night_mode'] = (signal[37:40] == '001')

    return command

# DEBUG
a = generate_ir_command(
    hvac_mode='heat',
    target_temp=27,
    fan_mode='auto',
    swing_mode='horizontal',
    turbo=False,
    night_mode=False
)

#d = parse_encoded_command(a)
#print(a == generate_ir_command(**d))
d = parse_encoded_command('B94MrAbTAe0EgAPgAwHgAw/gAx/gExfgBzPgCwvgPwHgK1/gGyfgS6vgR+/gB6s=')
#print("\n".join(f"{k}: {v}" for k, v in d.items()))

b = generate_ir_command(
    hvac_mode='heat',
    target_temp=27,
    fan_mode='auto',
    swing_mode='horizontal',
    power_on=False
)

#print(b)
