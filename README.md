<h1 align="center">Speaker Recognition Demo</h1>
<p align="center"><em>demonstration of speaker recognition using embeddings.</em></p>

## 🛠️ Prerequisites
- Python 3.x (tested with Python 3.11 via `pyenv`)
- Access to [Pyannote/embeddings](https://huggingface.co/pyannote/embedding) embeddings

#### Environment Variables
Create a `.env` file in the project root with the following keys:
```
GEMINI_API_KEY=<your_gemini_api_key>
HF_API_KEY=<your_huggingface_api_key>
```

#### Voice Samples
- Your voice sample should be stored as `.speakers/<your_name>.wav`
- Assistant voice sample should be stored as `.voices/<language_code>.wav`

## 📦 Installation
1. Clone the repository
```bash
git clone https://github.com/ecnivs/speaker_rec_demo.git
cd speaker_rec_demo
```
1. Install dependencies
```bash
pip install -r requirements.txt
```

## 🚀 Running
```bash
python main.py
```

## 🙌 Contributing
Feel free to:
1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Submit a pull request

#### *I'd appreciate any feedback or code reviews you might have!*
