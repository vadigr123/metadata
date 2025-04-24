# Metadata Editor by mikus

A "application" for viewing ~~and editing~~ metadata in images and LoRA model files.

## Features

- **Image Metadata Viewer**:
  - Display prompt, negative prompt, and generation parameters
  - Supports PNG and JPEG/JPG formats

- **LoRA Metadata Analyzer**:
  - View detailed training information from .safetensors files
  - Display model information, training parameters, dataset stats
  - Clean presentation of tag frequencies and suggested prompts

## Usage

1. **View Image Metadata**:
   - Drag and drop an image or click to browse
   - View all embedded metadata in organized sections

2. **Analyze LoRA Models**:
   - Drop a .safetensors file to view training metadata
   - See detailed information about the model and training process

## File Structure

```
metadata/
├── img/                    # Image assets
│   ├── civitai.png         # CivitAI icon
│   ├── discord.png         # Discord icon
│   ├── server.png          # Main server icon
├── index.html              # Main application file
├── script.js               # Application logic
├── style.css               # Stylesheets
└── README.md               # This file
```

## License

This project is licensed under the MIT License.

---

Created by:: [mikus (DeepSeek)](https://github.com/vadigr123) | [Discord](https://discord.gg/UtYvGwFfvx) | [CivitAI](https://civitai.com/user/vadigr123_) | [Telegram](https://t.me/ai_mikus)
