package org.deeplearning4j.examples.Alert1001;

import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.Alert1001DataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.RBM;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Collections;
/**
 * Created by nadsat on 11/05/2016.
 */
public class Alert1001Test {
    private static Logger log = LoggerFactory.getLogger(Alert1001Test.class);

    public static void main(String[] args) throws Exception {
        final int numRows = 12;
        final int numColumns = 1;
        int outputNum = 10;
        int numSamples = 60000;
        int batchSize = 100;
        int iterations = 10;
        int seed = 123;
        int listenerFreq = batchSize / 5;

        log.info("Load data....");
        DataSetIterator iter = new Alert1001DataSetIterator(batchSize,numSamples,true);

        log.info("Build model....");
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
            .seed(seed)
            .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)
            .gradientNormalizationThreshold(1.0)
            .iterations(iterations)
            .momentum(0.5)
            .momentumAfter(Collections.singletonMap(3, 0.9))
            .optimizationAlgo(OptimizationAlgorithm.CONJUGATE_GRADIENT)
            .list(4)
            .layer(0, new RBM.Builder().nIn(numRows*numColumns).nOut(500)
                .weightInit(WeightInit.XAVIER).lossFunction(LossFunction.RMSE_XENT)
                .visibleUnit(RBM.VisibleUnit.BINARY)
                .hiddenUnit(RBM.HiddenUnit.BINARY)
                .build())
            .layer(1, new RBM.Builder().nIn(500).nOut(250)
                .weightInit(WeightInit.XAVIER).lossFunction(LossFunction.RMSE_XENT)
                .visibleUnit(RBM.VisibleUnit.BINARY)
                .hiddenUnit(RBM.HiddenUnit.BINARY)
                .build())
            .layer(2, new RBM.Builder().nIn(250).nOut(200)
                .weightInit(WeightInit.XAVIER).lossFunction(LossFunction.RMSE_XENT)
                .visibleUnit(RBM.VisibleUnit.BINARY)
                .hiddenUnit(RBM.HiddenUnit.BINARY)
                .build())
            .layer(3, new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD).activation("softmax")
                .nIn(200).nOut(outputNum).build())
            .pretrain(true).backprop(false)
            .build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        model.setListeners(Collections.singletonList((IterationListener) new ScoreIterationListener(listenerFreq)));

        log.info("Train model....");
        model.fit(iter); // achieves end to end pre-training

        log.info("Evaluate model....");
        Evaluation eval = new Evaluation(outputNum);

        DataSetIterator testIter = new Alert1001DataSetIterator(100,10000);
        while(testIter.hasNext()) {
            DataSet testAlert1001 = testIter.next();
            INDArray predict2 = model.output(testAlert1001.getFeatureMatrix());
            eval.eval(testAlert1001.getLabels(), predict2);
        }

        log.info(eval.stats());
        log.info("****************Example finished********************");

    }

}
