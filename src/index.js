import dotenv from "dotenv"
import express from "express"
import path from "path"
import fs from "fs"
import jpeg from "jpeg-js"
//import * as tfjs from '@tensorflow/tfjs';
import * as tfjsnode from '@tensorflow/tfjs-node'

const PATH_MODEL = 'http://localhost:8081/mobilenet/model.json'
const IMAGE_TO_PREDICT = __dirname + '/static/img/pizza.jpg'

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
  const numberOfChannels = 4
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

  const outShape = [image.height, image.width, numberOfChannels]
  const input = tfjsnode.tensor3d(values, outShape, 'int32')
  input.shape.push(1)
  const model = await tfjsnode.loadLayersModel(PATH_MODEL)
  //console.log(model);
  //@TODO - ajustar a camada de input_1 para outshape correto
  model.predict(input);
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
  app.use(express.json())
  app.set("views", path.join( __dirname, "views"))
  app.set("view engine", "ejs")
  app.use(express.static( path.join( __dirname, "static")))

  app.listen(port, () => {
    callback(port)
  });

}

startServer((port)=>{
  console.log(`Servidor na url : http://localhost:${port}`)
  showNeuralNetWorkSummary()
  predictingImage(IMAGE_TO_PREDICT)
})