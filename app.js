// app.js
const express = require('express');
const multer = require('multer');
const path = require('path');
const fs = require('fs');
const cors = require('cors');
const { spawn } = require('child_process');

const app = express();
const PORT = 3000;

app.use(cors());
app.use(express.static(path.join(__dirname, 'public')));

const UPLOADS_DIR = path.join(__dirname, 'uploads');
if (!fs.existsSync(UPLOADS_DIR)) fs.mkdirSync(UPLOADS_DIR, { recursive: true });

const storage = multer.diskStorage({
  destination: (req, file, cb) => cb(null, UPLOADS_DIR),
  filename: (req, file, cb) => cb(null, `img_${Date.now()}${path.extname(file.originalname)}`)
});
const upload = multer({ storage });

// หน้าเว็บหลัก
app.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

// 🔥 ใช้โมเดลจริงในการทำนาย
app.post('/predict', upload.single('image'), (req, res) => {
  if (!req.file) return res.status(400).json({ error: 'No image uploaded' });

  const imagePath = req.file.path;
  const pythonProcess = spawn('python', ['predict_torch.py', imagePath]);

  let dataString = '';
  pythonProcess.stdout.on('data', data => dataString += data.toString());
  pythonProcess.stderr.on('data', err => console.error('Python Error:', err.toString()));

  pythonProcess.on('close', code => {
    try {
      const result = JSON.parse(dataString);
      fs.unlinkSync(imagePath); // ลบไฟล์หลังประมวลผล
      res.json(result);
    } catch (err) {
      console.error('Parse Error:', err);
      res.status(500).json({ error: 'Failed to parse prediction result.' });
    }
  });
});

app.listen(PORT, () => console.log(`Server running → http://localhost:${PORT}`));
