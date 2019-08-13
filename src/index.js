import * as tfjsnode from '@tensorflow/tfjs-node'
import { IMAGENET_CLASSES } from './imagenet_classes'

import dotenv from 'dotenv' 
import express from 'express'
import multer from 'multer'
import uuidv4 from 'uuid/v4'
import https from 'https'
import path from 'path'
import jimp from 'jimp';
import jpeg from 'jpeg-js'
import png from 'pngjs';
import bmp from 'bmp-js'
import fs from 'fs'

dotenv.config()

const {WIDTH_IMG, HEIGHT_IMG, NUM_OF_CHANNELS_RGB, NUM_OF_CHANNELS_RGBA, SERVER_PORT, GOOGLE_KEY, GOOGLE_CX} = process.env;
const PATH_MODEL = `http://localhost:${SERVER_PORT}/vgg19/model.json`
const GOOGLE_SEARCH_URL = `https://www.googleapis.com/customsearch/v1?key=${GOOGLE_KEY}&cx=${GOOGLE_CX}&q=`

/**
 * Detalha as camadas disponiveis no modelo de rede neural carregado.
 */
const showNeuralNetWorkSummary = async () => {
	const currentModel = await tfjsnode.loadLayersModel(PATH_MODEL)
	currentModel.summary();
}

/**
 * Escolhe a lib correta para decodificar a image e obtem seus dados no formato RGBA (Uint8Array).
 * @param {*} path - Caminho pra imagem
 * @param {*} mimetype - Formato da imagem
 * @returns {Promise}
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
 * Converte a matrix de com informações da imagem para 3 canais RGB (Int32Array).
 * @param {Array} rgbaDataImage 
 */
const convertImageToRGB = async (rgbaDataImage) => {

  return new Promise(async (resolve) => {
    const numRgbChannels = Number(NUM_OF_CHANNELS_RGB)
    const numRgbaChannels = Number(NUM_OF_CHANNELS_RGBA)
    const numPixels = Number(WIDTH_IMG) * Number(HEIGHT_IMG)
    const rgbDataImage = new Int32Array(numPixels * numRgbChannels)
    
    for (let i = 0; i < numPixels; i++) {
      for (let channel = 0; channel < numRgbChannels; ++channel) {
        rgbDataImage[i * numRgbChannels + channel] = rgbaDataImage[i * numRgbaChannels + channel]
      }
    }
    resolve(rgbDataImage)
  });
}

/**
 * Lê uma imagem do diretorio e carrega palavra de predição atrvés do reconhecimento pelo Tensorflow.
 * @param {*} dataImage - Data array que representa a imagem escolhida.
 * @returns {Promise}
 */
const predictingImage = async (rgbDataImage) => {

  return new Promise(async (resolve) => {
    const tensorArrayShape = [1, Number(HEIGHT_IMG), Number(WIDTH_IMG), Number(NUM_OF_CHANNELS_RGB)]
    //tfjsnode.tensor([1, 2, 3, 4]).print();
    let inputTensor = tfjsnode.tensor(rgbDataImage, tensorArrayShape, 'int32')
    let model = await tfjsnode.loadLayersModel(PATH_MODEL)
    inputTensor.print();
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
 * Realiza uma pesquisa pelo GoogleSearch e retorna uma Promise com JSON de resultados.
 * @param {*} query - Termo que será usado para pesquisa no Google.
 * @returns {Promise}
 */
const searchingOnGoogle = (query) => {

  return new Promise(async (resolve) => {
    let url = GOOGLE_SEARCH_URL+query
    https.get(url, res => {
      res.setEncoding('utf8');
      let body = '';
      res.on('data', data => {
        body += data;
      });
      res.on('end', () => {
        body = JSON.parse(body);
        resolve(body.items)
      });
    });
  });
}

/**
 * Inicia o servidor para upload de imagens
 * @returns {Promise}
 */
const startServer = async () => {

  return new Promise(async (resolve) => {
    const port = SERVER_PORT
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

/**
 * Contem as regras que serão executadas a cada requisição.
 * @param {*} app 
 */
const buildRouter = (app) => {

  app.get('*', (req, res) => {
      res.render('index', {image: undefined})
  });

  app.post('/upload', multer().single('photo'), (req, res) => {
    if(req.file) {
      let filename = uuidv4() + '.' +req.file.originalname.split('.').pop()
      let imagepath = __dirname + '/static/img/'+filename
      jimp.read(req.file.buffer).then(img => {
        img.resize(Number(WIDTH_IMG), Number(HEIGHT_IMG)).writeAsync(imagepath).then(() => {
          decodeImage(imagepath, req.file.mimetype).then(dataImage => {
            convertImageToRGB(dataImage).then(rgbDataImage => {
              predictingImage(rgbDataImage).then(predictions => {
                const [firstprediction] = predictions;
                searchingOnGoogle(firstprediction.class).then(results => {
                  res.render('index', { image: { 
                    data: req.file.buffer.toString('base64'), 
                    mimetype: req.file.mimetype,
                    predicitions: JSON.stringify(predictions, undefined, 2),
                    results: JSON.stringify(results, undefined, 2)
                    }
                  })
                })
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

startServer().then(port => {
  console.log(`Servidor na url : http://localhost:${port}`)
  //showNeuralNetWorkSummary()
})