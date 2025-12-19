// http://localhost:3000/api/data
// http://localhost:3000/api/update_machine
// http://localhost:3000/api/Login
// http://localhost:3000/api/register

// http://localhost/phpmyadmin/index.php?route=/&route=%2F
// 建立 MySQL 連線
const mysql = require('mysql2');
const express = require("express");
const cors = require("cors");
require('dotenv').config();

const app = express();
const port = 3000;
app.use(cors());
app.use(express.json());

const connection = mysql.createConnection({
  host: process.env.DB_HOST,
  port: process.env.DB_PORT || 3306,
  user: process.env.DB_USER,
  password: process.env.DB_PASSWORD,
  database: process.env.DB_NAME
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

// 登入帳號密碼
let USERS = [];
function loadUsers() {
  const sql = "SELECT username, password FROM user_base";
  
  connection.query(sql, (err, results) => {
    if (err) {
      console.error("查詢失敗:", err);
      return;
    }

    USERS = results.map(row => ({
      username: row.username,
      password: row.password
    }));

    console.log(`資料：${USERS}`)
  });
};
loadUsers();
app.post('/api/Login', (req, res) => {
    loadUsers();
    const { username, password ,ros_ip} = req.body;

    if(req.method !== 'POST'){
      return res.status(405).send('method not allowed');
    }

    const user = USERS.find(u => u.username === username && u.password === password);

    if(user){
      res.send({success:true, message:'登入成功', ros_ip:ros_ip})
    }else{
      res.send({ success: false, message: '帳號或密碼錯誤' });
    }
});

app.listen(port, () => {
  console.log(`Node.js 後端已啟動：http://localhost:${port}`);
});



// 註冊帳號 API
app.post('/api/register', (req, res) => {
    const { username, password } = req.body;

    if (!username || !password) {
        return res.status(400).json({
            success: false,
            message: '帳號或密碼錯誤'
        });
    }
    const defaultPermission = "9";

    const sql = 'INSERT INTO user_base (username, password, permissions) VALUES (?, ?, ?)';
    connection.query(sql, [username, password, defaultPermission], (err, result) => {
        if (err) {
            if (err.code === 'ER_DUP_ENTRY') {
                return res.json({
                    success: false,
                    message: '帳號已存在'
                });
            }
            
            console.error(err);
            return res.status(500).json({
                success: false,
                message: '註冊失敗（資料庫錯誤）'
            });
        }
        
        // 註冊成功
        res.json({
            success: true,
            message: '註冊成功'
        });
    });
});

