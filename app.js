const express = require('express');
const bodyParser = require('body-parser');
const fileUpload = require('express-fileupload');
const mongoose = require('mongoose');
const csv = require('fast-csv');
const tf = require('@tensorflow/tfjs');

const app = express();
const port = process.env.PORT || 3000;

app.use(bodyParser.urlencoded( { extended: true } ));
app.use(bodyParser.json());
app.use(fileUpload());

const healthRouter = express.Router();

healthRouter.route('/')
  .get((req, res) => {
    res.json({
      status: "Up!"
    });
  });

const SightingSchema = new mongoose.Schema({
  occurred_at: {type: String},
  city: {type: String},
  country: {type: String},
  shape: {type: String},
  duration_seconds: {type: String},
  description: {type: String},
  reported_on: {type: String},
  latitude: {type: String},
  longitude: {type: String}
});

const SightingModel = mongoose.model("sightings", SightingSchema);

const uploadRouter = express.Router();
uploadRouter.route('/')
  .post((req, res) => {
    if(req.files){
      let sightings = [];
      console.log("processing data...")
      csv.parseString(req.files.ufos.data.toString(), {headers: true, ignoreEmpty: true})
         .on("data", function(data){
           sightings.push(data);
         })
         .on("end", function(){
           SightingModel.create(sightings, function(err, docs){
             if (err) {
               throw err;
             }
             res.json({status: "Success! finished file"})
           })
           console.log(`success! stored ${sightings.length} sightings`);
         });
    }
    else{
      res.json({status: "no file"});
    }
  })

const trainRouter = express.Router();
trainRouter.route('/')
  .post((req, res) => {
    console.log("Got request", req.body);
    SightingModel.countDocuments()
      .then(count => {
        SightingModel
          .find()
          .then((documents) => {
            console.log("got the documents")
            const oneHot = outcome => Array.from(tf.oneHot(outcome, 2).dataSync());
            sightings_X = documents.map(document => {
              return ['duration_seconds', 'latitude', 'longitude'].map(feature => {
                const val = document[feature];
                return val === undefined ? 0 : parseFloat(val);
              });
            });
            sightings_y = documents.map(document => {
              // lets just assume lights are not a real ships and is a false alarm.
              const isShip = (document['shape'] === undefined || document['shape'] == "light") ? 0 : 1;
              return oneHot(isShip);
            })
            const dataSet = tf.data
              .zip({ xs: tf.data.array(sightings_X), ys: tf.data.array(sightings_y) })
              .shuffle(documents.length, 42);

            const splitIdx = parseInt((0.8) * documents.length, 10);

            const modelTrain = dataSet.take(splitIdx).batch(1000)
            const modelTest = dataSet.skip(splitIdx + 1).batch(1000)
            const validateTrain = tf.tensor(sightings_X.slice(splitIdx))
            const validateTest = tf.tensor(sightings_y.slice(splitIdx))

            const model = tf.sequential();
            model.add(
              tf.layers.dense({
                units:2,
                activation: "softmax",
                inputShape: 3
              })
            );

            const optimizer = tf.train.adam(0.001);
            model.compile({
              optimizer: optimizer,
              loss: "binaryCrossentropy",
              metrics: ["accuracy"]
            });
            console.log("training...")
            model.fitDataset(modelTrain, {
              epochs: 100,
              validationData: modelTest
            })

            const predReq = ['duration_seconds', 'latitude', 'longitude'].map(feature => {
              return parseFloat(req.body[feature]);
            });

            const predReqTensor = tf.reshape(tf.tensor(predReq), [1, 3]);
            console.log("making prediction....")
            const pred = model.predict(predReqTensor).argMax(-1);
            const output = pred.arraySync();
            // model.save('file://models')
            console.log("done!")
            res.json({ result: output });
          });
      })
  })

mongoose.set('useNewUrlParser', true);
mongoose.set('useFindAndModify', false);
mongoose.set('useCreateIndex', true);
mongoose.set('useUnifiedTopology', true);
mongoose.connect(process.env.DATABASE_CONNECTION)
  .then(() => {
    app.use('/health', healthRouter);
    app.use('/upload', uploadRouter);
    app.use('/train', trainRouter);
    app.listen(port, () => console.log(`Listening on ${port}. Health check at http://localhost:${port}/health`));
  })
