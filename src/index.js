import * as tfjsnode from '@tensorflow/tfjs-node'
import { IMAGENET_CLASSES } from './imagenet_classes'

import express from 'express'
import dotenv from 'dotenv'
import multer from 'multer'
import uuidv4 from 'uuid/v4'
import path from 'path'
import jimp from 'jimp';
import jpeg from 'jpeg-js'
import png from 'pngjs';
import bmp from 'bmp-js'
import fs from 'fs'

const PATH_MODEL = 'http://localhost:8081/vgg19/model.json'
const WIDTH_IMG = 224;
const HEIGHT_IMG = 224;
/**
 * Detalha as camadas disponiveis no modelo de rede neural carregado.
 */
const showNeuralNetWorkSummary = async () => {
	const currentModel = await tfjsnode.loadLayersModel(PATH_MODEL)
	currentModel.summary();
}

/**
 * Escolhe a lib correta para decodificar a image e obtem seus dados no formato ArrayInt32
 * @param {*} path - Caminho pra imagem
 * @param {*} mimetype - Formato da imagem
 */
const decodeImage = (path, mimetype) => {
  return new Promise((resolve, reject) => {
    const buffer = fs.readFileSync(path)
    let image
    switch (mimetype) {
      case 'image/jpeg':
        image = jpeg.decode(buffer, true)
        break;
      case 'image/png':
        image = png.PNG.sync.read(buffer)
        break;
      case 'image/bmp':
        image = bmp.decode(buffer)
        break;
      default:
          reject('error loading image')
    }
    resolve(image.data)
  });
}

/**
 * Lê uma imagem do diretorio e carrega palavra de predição atrvés do reconhecimento pelo Tensorflow
 * @param {*} path - Caminho pra imagem que será usada na predição
 */
const predictingImage = async (dataImage) => {
  return new Promise(async (resolve) => {
    const numberOfChannels = 3
    const numPixels = WIDTH_IMG * HEIGHT_IMG
    const values = new Int32Array(numPixels * numberOfChannels)
    const tensorArrayShape = [1, HEIGHT_IMG, WIDTH_IMG, numberOfChannels]

    for (let i = 0; i < numPixels; i++) {
      for (let channel = 0; channel < numberOfChannels; ++channel) {
        values[i * numberOfChannels + channel] = dataImage[i * 4 + channel]
      }
    }
    
    let inputTensor = tfjsnode.tensor(values, tensorArrayShape, 'int32')
    let model = await tfjsnode.loadLayersModel(PATH_MODEL)
    let predictions = model.predict(inputTensor).dataSync()
    let mappedProbalities = Array.from(predictions).map((p,i) => {
          return { probability: p, class: IMAGENET_CLASSES[i] };
    });
    let sortedProbalities = mappedProbalities.sort((a,b) => {
          return b.probability-a.probability;
    });
    resolve(sortedProbalities.slice(0,5));
  });
}

/**
 * Inicia o servidor para upload de imagens
 * @param {function} callback  - Função de callback
 */
const startServer = async () => {
  return new Promise(async (resolve) => {
    dotenv.config()
    const port = process.env.SERVER_PORT
    const app = express()
    app.use(express.json())
    app.set('views', path.join( __dirname, 'views'))
    app.set('view engine', 'ejs')
    app.use(express.static( path.join( __dirname, 'static')))
    app.listen(port, () => {
      buildRouter(app)
      resolve(port)
    });
  });
}

const buildRouter = (app) => {
  app.get('*', (req, res) => {
      res.render('index', {image: undefined})
  });

  app.post('/upload', multer().single('photo'), (req, res) => {
    if(req.file) {
      let filename = uuidv4() + '.' +req.file.originalname.split('.').pop()
      let imagepath = __dirname + '/static/img/'+filename
      jimp.read(req.file.buffer).then(img => {
        img.resize(WIDTH_IMG, HEIGHT_IMG).writeAsync(imagepath).then(() => {
          decodeImage(imagepath, req.file.mimetype).then(dataImage => {
            predictingImage(dataImage).then(predictions => {
              res.render('index', { image: { 
                  data: req.file.buffer.toString('base64'), 
                  mimetype: req.file.mimetype,
                  predicitions: JSON.stringify(predictions, undefined, 2)
                }
              })
            })
          })
        })
      })
      .catch(err => {
        console.error(err)
      });
    }
    else throw 'error upload';
  });
}

startServer().then(data => {
  console.log(`Servidor na url : http://localhost:${data}`)
  showNeuralNetWorkSummary()
})