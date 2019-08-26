package com.yptheangel.dl4jandroid.yolo_objdetection.utils;

import android.os.Environment;
import android.util.Log;

import org.deeplearning4j.zoo.util.ClassPrediction;
import org.deeplearning4j.zoo.util.Labels;
import org.nd4j.base.Preconditions;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.BufferedInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;

public abstract class BaseLabelsAndroid implements Labels {

    protected ArrayList<String> labels;
    String LOG_TAG="DEMO_BaseLabelsAndroid";

    protected BaseLabelsAndroid() throws IOException {
        this.labels = getLabels();
    }

    public ArrayList<String> getLabels () throws IOException {

        ArrayList<String> labels = new ArrayList<>();
        File labelFile = new File(Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS), "voc.names");
        Log.i(LOG_TAG,"LabelFile located at : "+labelFile);

        try (InputStream is = new BufferedInputStream(new FileInputStream(labelFile)); Scanner s = new Scanner(is)) {
            while (s.hasNextLine()) {
                labels.add(s.nextLine());
            }
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }
        return labels;
    }

    public String getLabel(int n) {
        Preconditions.checkArgument(n >= 0 && n < this.labels.size(), "Invalid index: %s. Must be in range0 <= n < %s", n, this.labels.size());
        return (String)this.labels.get(n);
    }

    public List<List<ClassPrediction>> decodePredictions(INDArray predictions, int n) {
        if (predictions.rank() == 1) {
            predictions = predictions.reshape(1L, predictions.length());
        }

        Preconditions.checkState(predictions.size(1) == (long)this.labels.size(), "Invalid input array: expected array with size(1) equal to numLabels (%s), got array with shape %s", this.labels.size(), predictions.shape());
        int rows = (int)predictions.size(0);
        int cols = (int)predictions.size(1);
        if (predictions.isColumnVectorOrScalar()) {
            predictions = predictions.ravel();
            rows = (int)predictions.size(0);
            cols = (int)predictions.size(1);
        }

        List<List<ClassPrediction>> descriptions = new ArrayList();

        for(int batch = 0; batch < rows; ++batch) {
            INDArray result = predictions.getRow((long)batch);
            result = Nd4j.vstack(new INDArray[]{Nd4j.linspace(0L, (long)cols, (long)cols), result});
            result = Nd4j.sortColumns(result, 1, false);
            List<ClassPrediction> current = new ArrayList();

            for(int i = 0; i < n; ++i) {
                int label = result.getInt(new int[]{0, i});
                double prob = result.getDouble(1L, (long)i);
                current.add(new ClassPrediction(label, this.getLabel(label), prob));
            }

            descriptions.add(current);
        }

        return descriptions;
    }
}
