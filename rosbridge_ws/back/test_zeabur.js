// http://goof:3000/api/data
// http://goof:3000/api/update_machine
// http://goof:8080/api/Login
// http://goof:3000/api/register
// 建立 MySQL 連線
const mysql = require('mysql2/promise');
const express = require("express");
const cors = require("cors");
require('dotenv').config();

const app = express();
const port = process.env.PORT || 3000;
app.use(cors());
app.use(express.json());

const pool = mysql.createPool({
  host: process.env.DB_HOST,
  port: Number(process.env.DB_PORT) || 3306,
  user: process.env.DB_USER,
  password: process.env.DB_PASSWORD,
  database: process.env.DB_NAME,

  waitForConnections: true,
  connectionLimit: 10,
  queueLimit: 0,
});

// 測試連線
pool.getConnection()
  .then(connection => {
    console.log("✅ 已連接 Zeabur 資料庫");
    connection.release();
  })
  .catch(err => {
    console.error("❌ 資料庫連線失敗:", err.message);
  });



// 連接前端初始化生成
app.get("/api/data", async (req, res) => {
  try {
    const connection = await pool.getConnection();
    const [results] = await connection.query("SELECT * FROM car_state");
    connection.release();
    console.log(`資料：`, results);
    res.json(results);
  } catch (err) {
    console.error("查詢失敗:", err);
    res.status(500).json({ error: err.message });
  }
});

// 更新機台資料 API
app.post('/api/update_machine', async (req, res) => {
    try {
      const { car_number, line_speed, angle_speed } = req.body;
      const now = new Date().toLocaleString('zh-TW', { hour12: false }).replace('/', '-').replace('/', '-');
      const sql = 'UPDATE car_state SET line_speed = ?, angle_speed = ?, local_time = ? WHERE car_number = ?';
      
      const connection = await pool.getConnection();
      const [result] = await connection.query(sql, [line_speed, angle_speed, now, car_number]);
      connection.release();
      
      res.json({ success: true, message: '更新成功', result });
      console.log('更新資料:', { car_number, line_speed, angle_speed, now });
    } catch (err) {
      console.error(err);
      res.status(500).json({ success: false, message: '更新失敗' });
    }
});

// 登入帳號密碼
let USERS = [];

async function loadUsers() {
  try {
    const connection = await pool.getConnection();
    const [results] = await connection.query("SELECT username, password FROM user_base");
    connection.release();
    
    USERS = results.map(row => ({
      username: row.username,
      password: row.password
    }));
    
    console.log(`資料：`, USERS);
  } catch (err) {
    console.error("查詢失敗:", err);
  }
}

loadUsers();

app.post('/api/Login', async (req, res) => {
    await loadUsers();
    const { username, password, ros_ip } = req.body;

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
  console.log("MYSQL_HOST:", process.env.MYSQL_HOST);
  console.log("MYSQL_PORT:", process.env.MYSQL_PORT);
  console.log("MYSQL_USER:", process.env.MYSQL_USER);
  console.log("MYSQL_PASSWORD:", process.env.MYSQL_PASSWORD);
  console.log("MYSQL_DATABASE:", process.env.MYSQL_DATABASE);
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
    pool.query(sql, [username, password, defaultPermission], (err, result) => {
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