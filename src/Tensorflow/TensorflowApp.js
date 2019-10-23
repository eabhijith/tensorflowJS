import React,{Component} from "react"
import * as tf from '@tensorflow/tfjs'


export default class TensorflowApp extends Component{

    
    state={
        predictedValue:0
    }


    handlePredictValue = async () => {
        const model = tf.sequential(),
              xs = tf.tensor2d([-1,0,1,2,3,4],[6,1]),
              ys = tf.tensor2d([-3,-1,1,3,5,7],[6,1]);
        let predictedValue = 0;
        model.add(tf.layers.dense({units:1, inputShape: [1]}));
        model.compile({
            loss: 'meanSquaredError',
            optimizer: 'sgd'

        });

        await model.fit(xs,ys,{epochs:250});
        predictedValue = model.predict(tf.tensor2d([0],[1,1]));

        predictedValue = model.predict(tf.tensor2d([6],[1,1]));
        this.setState(
            {
                predictedValue: predictedValue.dataSync()[0]
            }
        );
    }  

    render(){
        return(
            <div>
            <header className="App-header">
            <button onClick={this.handlePredictValue}>Predict Value</button> <br/>
            <p>Predicted Value is: {this.state.predictedValue}</p>
            </header>
            </div>
                
        );
    }
}
