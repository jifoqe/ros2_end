// // 建立 MySQL 連線池（使用 Promise 版本以支援現代 async/await）
// // 針對 Zeabur serverless 環境優化
// const mysql = require('mysql2/promise');
// const express = require("express");
// const cors = require("cors");
// require('dotenv').config();

// const app = express();
// const port = 3000;
// app.use(cors());
// app.use(express.json());

// // 建立連線池設定 - 針對 serverless 環境優化
// const poolConfig = {
//   host: process.env.DB_HOST,
//   port: process.env.DB_PORT || 3306,
//   user: process.env.DB_USER,
//   password: process.env.DB_PASSWORD,
//   database: process.env.DB_NAME,
//   waitForConnections: true,
//   connectionLimit: 3,           // 降低連線數以適應 serverless
//   queueLimit: 0,
//   enableKeepAlive: true,
//   keepAliveInitialDelayMs: 10000,
//   connectTimeout: 10000,        // 連線超時
//   maxIdle: 3,                   // 最大閒置連線數
//   idleTimeout: 30000,           // 30秒閒置超時
// };

// let pool = null;
// let isPoolHealthy = false;

// // 建立新的連線池
// async function createPool() {
//   try {
//     // 如果舊池存在，先嘗試關閉
//     if (pool) {
//       try {
//         await pool.end();
//       } catch (e) {
//         console.warn('[DB] 關閉舊連線池時發生錯誤:', e.message);
//       }
//     }

//     pool = mysql.createPool(poolConfig);
//     isPoolHealthy = true;
//     console.log('[DB] MySQL 連線池已建立');
//     return pool;
//   } catch (err) {
//     console.error('[DB] MySQL 連線池建立失敗:', err.message);
//     isPoolHealthy = false;
//     throw err;
//   }
// }

// // 取得健康的連線
// async function getConnection() {
//   // 如果連線池不存在或不健康，重新建立
//   if (!pool || !isPoolHealthy) {
//     console.log('[DB] 連線池需要重新建立...');
//     await createPool();
//   }

//   try {
//     const connection = await pool.getConnection();
//     // 測試連線是否有效
//     await connection.ping();
//     return connection;
//   } catch (err) {
//     console.error('[DB] 取得連線失敗:', err.message);
//     isPoolHealthy = false;

//     // 連線失敗，嘗試重建連線池並重新取得連線
//     if (err.message.includes('closed state') ||
//       err.message.includes('ECONNREFUSED') ||
//       err.message.includes('ETIMEDOUT') ||
//       err.message.includes('PROTOCOL_CONNECTION_LOST')) {
//       console.log('[DB] 嘗試重建連線池...');
//       await createPool();
//       const connection = await pool.getConnection();
//       await connection.ping();
//       return connection;
//     }
//     throw err;
//   }
// }

// // 安全執行查詢的包裝函數
// async function executeQuery(queryFn) {
//   let connection;
//   try {
//     connection = await getConnection();
//     const result = await queryFn(connection);
//     return result;
//   } finally {
//     if (connection) {
//       try {
//         connection.release();
//       } catch (e) {
//         console.warn('[DB] 釋放連線時發生錯誤:', e.message);
//       }
//     }
//   }
// }

// // 初始化連線池
// async function initializePool() {
//   try {
//     await createPool();
//     const connection = await getConnection();
//     connection.release();
//     console.log('[DB] MySQL 連線池已初始化並驗證成功');
//   } catch (err) {
//     console.error('[DB] MySQL 連線池初始化失敗:', err.message);
//     // 在 serverless 環境中，不需要重試，會在請求時重新建立
//     isPoolHealthy = false;
//   }
// }

// // 啟動應用時初始化池
// initializePool();

// // 連線前端初始化生成
// app.get("/api/data", async (req, res) => {
//   try {
//     const results = await executeQuery(async (connection) => {
//       const [rows] = await connection.query("SELECT * FROM car_state");
//       return rows;
//     });
//     console.log(`[API] /data - 查詢成功，取得 ${results.length} 筆資料`);
//     res.json(results);
//   } catch (err) {
//     console.error("[API] /data - 查詢失敗:", err.message);
//     res.status(500).json({ error: err.message });
//   }
// });

// // 更新機台資料 API
// app.post('/api/update_machine', async (req, res) => {
//   try {
//     const { car_number, line_speed, angle_speed } = req.body;
//     const now = new Date().toLocaleString('zh-TW', { hour12: false }).replace('/', '-').replace('/', '-');

//     const result = await executeQuery(async (connection) => {
//       const [rows] = await connection.query(
//         'UPDATE car_state SET line_speed = ?, angle_speed = ?, local_time = ? WHERE car_number = ?',
//         [line_speed, angle_speed, now, car_number]
//       );
//       return rows;
//     });

//     console.log('[API] /update_machine - 更新成功:', { car_number, line_speed, angle_speed, now });
//     res.json({ success: true, message: '更新成功', result });
//   } catch (err) {
//     console.error('[API] /update_machine - 更新失敗:', err.message);
//     res.status(500).json({ success: false, message: '更新失敗: ' + err.message });
//   }
// });

// // 登入帳號密碼
// let USERS = [];

// async function loadUsers() {
//   try {
//     const results = await executeQuery(async (connection) => {
//       const [rows] = await connection.query("SELECT username, password FROM user_base");
//       return rows;
//     });

//     USERS = results.map(row => ({
//       username: row.username,
//       password: row.password
//     }));

//     console.log(`[DB] 已載入 ${USERS.length} 位用戶`);
//   } catch (err) {
//     console.error('[DB] 載入用戶失敗:', err.message);
//   }
// }

// // 定期重新載入用戶（每60秒，serverless 環境降低頻率）
// setInterval(loadUsers, 60000);

// // 延遲載入用戶，避免與初始化衝突
// setTimeout(loadUsers, 3000);

// app.post('/api/Login', (req, res) => {
//   const { username, password, ros_ip } = req.body;

//   if (req.method !== 'POST') {
//     return res.status(405).send('method not allowed');
//   }

//   const user = USERS.find(u => u.username === username && u.password === password);

//   if (user) {
//     console.log('[API] /Login - 登入成功:', username);
//     res.send({ success: true, message: '登入成功', ros_ip: ros_ip })
//   } else {
//     console.log('[API] /Login - 登入失敗:', username);
//     res.send({ success: false, message: '帳號或密碼錯誤' });
//   }
// });

// app.listen(port, () => {
//   console.log(`Node.js 後端已啟動：http://localhost:${port}`);
// });



// // 註冊帳號 API
// app.post('/api/register', (req, res) => {
//     const { username, password } = req.body;

//     if (!username || !password) {
//         return res.status(400).json({
//             success: false,
//             message: '帳號或密碼錯誤'
//         });
//     }
//     const defaultPermission = "9";

//     const sql = 'INSERT INTO user_base (username, password, permissions) VALUES (?, ?, ?)';
//     connection.query(sql, [username, password, defaultPermission], (err, result) => {
//         if (err) {
//             if (err.code === 'ER_DUP_ENTRY') {
//                 return res.json({
//                     success: false,
//                     message: '帳號已存在'
//                 });
//             }
            
//             console.error(err);
//             return res.status(500).json({
//                 success: false,
//                 message: '註冊失敗（資料庫錯誤）'
//             });
//         }
        
//         // 註冊成功
//         res.json({
//             success: true,
//             message: '註冊成功'
//         });
//     });
// });






























require('dotenv').config(); // 載入環境變數
const mysql = require('mysql2/promise'); // 使用 Promise 版本以支援 async/await
const express = require("express");
const cors = require("cors");

const app = express();
const port = process.env.PORT || 3000;

app.use(cors());
app.use(express.json());

// 1. 使用連線池 (Connection Pool)
const pool = mysql.createPool({
  host: process.env.DB_HOST,
  port: process.env.DB_PORT,
  user: process.env.DB_USER,
  password: process.env.DB_PASSWORD,
  database: process.env.DB_NAME,
  waitForConnections: true,
  connectionLimit: 10,
  queueLimit: 0
});

// 檢查連線狀態
(async () => {
  try {
    const connection = await pool.getConnection();
    console.log('✅ 已成功連線至 Zeabur MySQL');
    connection.release();
  } catch (err) {
    console.error('❌ 資料庫連線失敗:', err.message);
  }
})();

// --- API 路由 ---

// 獲取機台資料
app.get("/api/data", async (req, res) => {
  try {
    const [rows] = await pool.query("SELECT * FROM car_state");
    res.json(rows);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// 更新機台資料
app.post('/api/update_machine', async (req, res) => {
  const { car_number, line_speed, angle_speed } = req.body;
  // 建議使用資料庫自帶的 NOW() 處理時間，更精準
  const sql = 'UPDATE car_state SET line_speed = ?, angle_speed = ?, local_time = NOW() WHERE car_number = ?';
  
  try {
    const [result] = await pool.query(sql, [line_speed, angle_speed, car_number]);
    res.json({ success: true, message: '更新成功', result });
  } catch (err) {
    res.status(500).json({ success: false, message: '更新失敗' });
  }
});

// 登入邏輯 (優化：直接查詢資料庫，不預載到記憶體，這對多使用者更安全且準確)
app.post('/api/Login', async (req, res) => {
  const { username, password, ros_ip } = req.body;

  try {
    const sql = "SELECT * FROM user_base WHERE username = ? AND password = ?";
    const [rows] = await pool.query(sql, [username, password]);

    if (rows.length > 0) {
      res.send({ success: true, message: '登入成功', ros_ip: ros_ip });
    } else {
      res.status(401).send({ success: false, message: '帳號或密碼錯誤' });
    }
  } catch (err) {
    res.status(500).send({ success: false, message: '伺服器錯誤' });
  }
});

// 註冊帳號
app.post('/api/register', async (req, res) => {
  const { username, password } = req.body;

  if (!username || !password) {
    return res.status(400).json({ success: false, message: '請輸入帳號密碼' });
  }

  const defaultPermission = "9";
  const sql = 'INSERT INTO user_base (username, password, permissions) VALUES (?, ?, ?)';

  try {
    await pool.query(sql, [username, password, defaultPermission]);
    res.json({ success: true, message: '註冊成功' });
  } catch (err) {
    if (err.code === 'ER_DUP_ENTRY') {
      return res.json({ success: false, message: '帳號已存在' });
    }
    res.status(500).json({ success: false, message: '註冊失敗' });
  }
});

app.listen(port, () => {
  console.log(`🚀 後端伺服器運行中：http://localhost:${port}`);
});