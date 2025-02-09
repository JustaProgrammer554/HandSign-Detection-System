import express from "express";
import bodyParser from "body-parser";

const app = express();
const port = 4000;

app.use(bodyParser.urlencoded({extended: true}));
app.use(express.static("public")); // Static files from the "public" folder.

// Define a route
app.get("/", (req, res) => {
    res.render("index.ejs");
});

app.get("/views/about.ejs", (req, res) => {
    res.render("about.ejs");
});

// start a server on port 3000
app.listen(port, () => {
    console.log(`listening on port ${port}`);
})