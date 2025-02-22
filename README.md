# Osaka AC Integration for Home Assistant

Osaka AC CSU-09HHAA integration for Home Assistant.
Complete remote replacement to control your AC from Home Assistant.

Osaka CSU-09HHAA is IR controlled air conditioner with both heat and cool modes together with some additional modes such as fan only, dry and "feel" (basically - auto) modes.

<img src="https://raw.githubusercontent.com/Astro1247/osaka_integration/master/docs/media/card-preview.png" width="50%">

## Features

- Fully replicates all original remote features
- Saves all settings between Home Assistant reboots

### Replicated features from original remote

#### Heat mode

- 16-31 degrees celsius temperature range
- Boost (turbo) mode (AC will lock fan speed to max and temperature to 31)

#### Cool mode

- 16-31 degrees celsius temperature range
- Boost (turbo) mode (AC will lock fan speed to max and temperature to 16)

#### Available in any mode, including Dry / Fan / Feel (Auto) modes

- Toggle power
- Low, Medium, High, Auto fan speeds
- Sleep (night) mode (AC will lock fan speed to quite mode)
- Swing positions (modes), including horizontal, 30 Degrees, Diagonal, 60 Degrees, Vertical, Swing and Auto

## Supported devices

### AC

- **Osaka**
  - `CSU-09HHAA`

### IR Blaster

- **Moes**
  - `UFO-R11` 

Probably any Zigbee2MQTT compatible IR blaster, but tested only with this one.

*There are chances that it can work with some similar AC models too, but it cannot be guaranteed and should be carefully tested before using with other models!*

## Installation

### Via [Home Assistant Community Store](https://hacs.xyz/docs/setup/download/) (HACS) add-in
<a href="https://my.home-assistant.io/redirect/hacs_repository/?owner=Astro1247&repository=osaka_integration&category=integration" target="_blank"><img src="https://my.home-assistant.io/badges/hacs_repository.svg" alt="Open your Home Assistant instance and open a repository inside the Home Assistant Community Store." /></a>

## Configuration

Declare your climate entity inside Home Assistant `configuration.yaml` file as follows:

```yaml
climate:
  - platform: osaka_ac
  name: Osaka AC
  unique_id: "osaka_ac"
```

No additional configuration is needed.

## How to use

Integration is compatible with all default climate cards, including Mushroom Climate Card.

### With [Mushroom Climate Card](https://github.com/piitaya/lovelace-mushroom/blob/main/docs/cards/climate.md)

```yaml
type: custom:mushroom-climate-card
entity: climate.osaka_ac
fill_container: false
show_temperature_control: true
hvac_modes:
  - "off"
  - heat
  - fan_only
  - cool
  - dry
  - auto
collapsible_controls: false
```

## To Do

- [@mildsunrise](https://github.com/mildsunrise) in [Documentation of Tuya's compression scheme for IR codes](https://gist.github.com/mildsunrise/1d576669b63a260d2cff35fda63ec0b5) mentioned [FastLZ](https://ariya.github.io/FastLZ), probably IR encoding and decoding should be reworked to use FastLZ.

## Contributing

Contributions are welcome!

## Thanks To

 - [Documentation of Tuya's compression scheme for IR codes](https://gist.github.com/mildsunrise/1d576669b63a260d2cff35fda63ec0b5) by [@mildsunrise](https://github.com/mildsunrise)