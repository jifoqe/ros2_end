// http://localhost:3000/api/data
// http://localhost:3000/api/update_machine
// 建立 MySQL 連線
const mysql = require('mysql2');
const express = require("express");
const cors = require("cors");

const app = express();
const port = 3000;
app.use(cors());
app.use(express.json());

const connection = mysql.createConnection({
  host: 'localhost',
  user: 'jerry',
  password: '000000',
  database: 'esp32'
});

connection.connect(err => {
  if (err) throw err;
  console.log('已連接 MySQL');
});

// 連接前端初始化生成
app.get("/api/data", (req, res) => {
  const sql = "SELECT * FROM car_state";
  
  connection.query(sql, (err, results) => {
    if (err) {
      console.error("查詢失敗:", err);
      return res.status(500).json({ error: err.message });
    }
    console.log(`資料：${results}`)
    res.json(results);
  });
});

// 更新機台資料 API
app.post('/api/update_machine', (req, res) => {
    const { car_number, line_speed, angle_speed } = req.body;
    const now = new Date().toLocaleString('zh-TW', { hour12: false }).replace('/', '-').replace('/', '-');
    const sql = 'UPDATE car_state SET line_speed = ?, angle_speed = ?, local_time = ? WHERE car_number = ?';
    
    connection.query(sql, [line_speed, angle_speed, now, car_number], (err, result) => {
        if (err) {
            console.error(err);
            return res.status(500).json({ success: false, message: '更新失敗' });
        }
        res.json({ success: true, message: '更新成功', result });
        console.log('更新資料:', { car_number, line_speed, angle_speed, now });
    });
});

app.listen(port, () => {
  console.log(`Node.js 後端已啟動：http://localhost:${port}`);
});
