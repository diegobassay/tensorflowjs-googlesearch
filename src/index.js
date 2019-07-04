import dotenv from "dotenv"
import express from "express"
import path from "path"
import fs from "fs"
import jpeg from "jpeg-js"
import * as tfjsnode from "@tensorflow/tfjs-node"
import { IMAGENET_CLASSES } from "./imagenet_classes";


const PATH_MODEL = 'http://localhost:8081/vgg19/model.json'
const IMAGE_TO_PREDICT = __dirname + '/static/img/beijaflor.jpg'

/**
 * Detalha as camadas disponiveis no modelo de rede neural carregado.
 */
const showNeuralNetWorkSummary = async () => {
	const currentModel = await tfjsnode.loadLayersModel(PATH_MODEL)
	currentModel.summary();
}

/**
 * Lê uma imagem do diretorio e carrega palavra de predição atrvés do reconhecimento pelo Tensorflow
 * @param {*} path - Caminho pra imagem que será usada na predição
 */
const predictingImage = async (path) => {
  const numberOfChannels = 3
  const buffer = fs.readFileSync(path)
  const image = jpeg.decode(buffer, true)
  const pixels = image.data
  const numPixels = image.width * image.height
  const values = new Int32Array(numPixels * numberOfChannels)

  for (let i = 0; i < numPixels; i++) {
    for (let channel = 0; channel < numberOfChannels; ++channel) {
      values[i * numberOfChannels + channel] = pixels[i * 4 + channel]
    }
  }

  const arrayShape = [1, image.height, image.width, numberOfChannels]
  const inputTensor = tfjsnode.tensor(values, arrayShape, "int32")
  const model = await tfjsnode.loadLayersModel(PATH_MODEL)
  let predictions = model.predict(inputTensor).dataSync();

  let mappedProbalities = Array.from(predictions).map((p,i) => {
        return { probability: p, class: IMAGENET_CLASSES[i] };
  });

  let sortedProbalities = mappedProbalities.sort(function(a,b){
        return b.probability-a.probability;
  });

  console.log(sortedProbalities.slice(0,5));
}
/**
 * Inicia o servidor para upload de imagens
 * @todo necessário configurar o middleware para receber upload e terminar rota para template ejs.
 * @param {function} callback  - Função de callback
 */
const startServer = async (callback) => {
    dotenv.config()
    const port = process.env.SERVER_PORT
    const app = express()
    const router = express.Router();
    
    router.get('/', async function (req, res) {
        await res.render('index');
    });

    app.use(express.json())
    app.set("views", path.join( __dirname, "views"))
    app.set("view engine", "ejs")
    app.use(express.static( path.join( __dirname, "static")))
    app.use('/', router);
    app.listen(port, () => {
        callback(port)
    });
}

startServer((port)=>{
  console.log(`Servidor na url : http://localhost:${port}`)
  //showNeuralNetWorkSummary()
  predictingImage(IMAGE_TO_PREDICT)
})