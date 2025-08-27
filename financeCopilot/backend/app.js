const mongoose = require("mongoose");
const express = require("express");
const cors = require("cors");
const session = require("express-session");
const bodyParser = require("body-parser");
const mongodbStore = require("connect-mongodb-session")(session);
const mongodbAPI =
  "mongodb+srv://fempsoft:yjJCf6DA9aMTDrvR@financecopilot.oeh7dgr.mongodb.net/";
const swaggerUi = require("swagger-ui-express");
const swaggerJsdoc = require("swagger-jsdoc");

const swaggerOptions = {
  swaggerDefinition: {
    openapi: "3.0.0",
    info: {
      title: "financeCopilot API",
      version: "1.0.0",
      description: "API documentation for financeCopilot",
      contact: {
        name: "fempsoft",
        url: "https://github.com/ahmetefeersoy/Finance_Copilot/tree/main",
        email: "fempsoft@gmail.com",
      },
    },
    servers: [
      {
        url: "http://localhost:8080",
      },
    ],
  },
  apis: ["./src/routes/*.js"],
};

const swaggerSpecs = swaggerJsdoc(swaggerOptions);

const app = express();
const store = new mongodbStore({
  uri: mongodbAPI,
  collection: "sessions",
  connectionOptions: {
    tls: true,
    tlsAllowInvalidCertificates: true,
    tlsAllowInvalidHostnames: true,
  },
});

const loginRoute = require("./src/routes/login.js");
const userPanelRoute = require("./src/routes/userPanel.js");
const generatorRoute = require("./src/routes/generator.js");
const copilotRoute = require("./src/routes/copilot.js");
const transactionsRoute = require("./src/routes/transactions.js");
require("./src/services/stockSocket.js");
app.use(bodyParser.urlencoded({ extended: false }));

const corsOptions = {
  origin: "http://localhost:3000",
  credentials: true,
};
app.use(cors(corsOptions));

app.use(bodyParser.urlencoded({ extended: false }));
app.use(bodyParser.json());
app.use(express.json());

app.use(
  session({
    secret: "fempsoft",
    resave: false,
    saveUninitialized: false,
    store: store,
    cookie: {
      maxAge: 24 * 60 * 60 * 1000,
      httpOnly: true,
      secure: false,
      sameSite: "lax",
    },
  })
);

app.get("/", (req, res) => {
  res.redirect("/api-docs");
});

app.use("/userPanel", userPanelRoute);
app.use("/generator", generatorRoute);
app.use("/login", loginRoute);
app.use("/copilot", copilotRoute);
app.use("/transactions", transactionsRoute);
app.use("/api-docs", swaggerUi.serve, swaggerUi.setup(swaggerSpecs));

mongoose
  .connect(mongodbAPI, {
    ssl: true,
    tlsAllowInvalidCertificates: true,
    tlsAllowInvalidHostnames: true,
  })
  .then((result) => {
    app.listen(8080, () => console.log("Server is running"));
  })
  .catch((err) => console.error("MongoDB connection errors:", err));
