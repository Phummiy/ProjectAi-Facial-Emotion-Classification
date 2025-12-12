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

// Serve static folder สำหรับไฟล์ heatmap
app.use('/static', express.static(path.join(__dirname, 'static')));

// โฟลเดอร์อัปโหลดชั่วคราว
const UPLOADS_DIR = path.join(__dirname, 'uploads');
if (!fs.existsSync(UPLOADS_DIR)) fs.mkdirSync(UPLOADS_DIR, { recursive: true });

const storage = multer.diskStorage({
  destination: (req, file, cb) => cb(null, UPLOADS_DIR),
  filename: (req, file, cb) => cb(null, `img_${Date.now()}${path.extname(file.originalname)}`)
});
const upload = multer({ storage });

// -----------------------------
// ลบ heatmap เก่าที่เกิน 15 นาที
// -----------------------------
const HEATMAP_DIR = path.join(__dirname, 'static', 'heatmaps');
const MAX_FILE_AGE = 15 * 60 * 1000; // 15 นาทีเป็นมิลลิวินาที

function cleanOldHeatmaps() {
  fs.readdir(HEATMAP_DIR, (err, files) => {
    if (err) return console.error('Error reading heatmaps folder:', err);
    const now = Date.now();
    files.forEach(file => {
      const filePath = path.join(HEATMAP_DIR, file);
      fs.stat(filePath, (err, stats) => {
        if (err) return console.error('Error getting file stats:', err);
        const age = now - stats.mtimeMs; // mtimeMs = last modified
        if (age > MAX_FILE_AGE) {
          fs.unlink(filePath, err => {
            if (err) console.error('Failed to delete old heatmap:', err);
            else console.log('Deleted old heatmap:', file);
          });
        }
      });
    });
  });
}

// เรียก cleanOldHeatmaps ทุก ๆ 1 นาที
setInterval(cleanOldHeatmaps, 60 * 1000);

// หน้าเว็บหลัก
app.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

// POST /predict
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

    // ลบรูปต้นฉบับหลังประมวลผล
    fs.unlinkSync(imagePath);

    // ลบ heatmap เก่า
    cleanOldHeatmaps();

    // -------------------------
    // บันทึก prediction ลงไฟล์เดียว
    // -------------------------
    const SAVE_DIR = path.join(__dirname, 'saved_predictions');
    if (!fs.existsSync(SAVE_DIR)) {
      fs.mkdirSync(SAVE_DIR, { recursive: true });
    }

    const jsonPath = path.join(SAVE_DIR, 'predictions.json');

    // อ่านไฟล์เดิม (ถ้ามี)
    let oldData = [];
    if (fs.existsSync(jsonPath)) {
      try {
        oldData = JSON.parse(fs.readFileSync(jsonPath, 'utf8'));
      } catch (err) {
        console.error("⚠ อ่าน predictions.json ไม่ได้, จะสร้างใหม่");
      }
    }

    // สร้าง object ใหม่ที่จะบันทึก
    const newEntry = {
      timestamp: new Date().toISOString(),
      image_name: req.file.originalname,
      result
    };

    // append
    oldData.push(newEntry);

    // เขียนกลับลงไฟล์
    fs.writeFileSync(jsonPath, JSON.stringify(oldData, null, 2));

    console.log("✔ Updated predictions.json");

    // ส่งกลับไปพร้อมชื่อไฟล์
    res.json({
      ...result,
      saved_file: "predictions.json"
    });

  } catch (err) {
    console.error('Parse Error:', err);
    res.status(500).json({ error: 'Failed to parse prediction result.' });
  }
});


});

// เริ่ม server
app.listen(PORT, () => console.log(`Server running → http://localhost:${PORT}`));
