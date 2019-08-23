package com.yptheangel.dl4jandroid.yolo_objdetection.utils;

import android.os.Environment;
import android.util.Log;

import org.deeplearning4j.zoo.util.ClassPrediction;
import org.deeplearning4j.zoo.util.Labels;
import org.nd4j.linalg.api.ndarray.INDArray;

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

    @Override
    public String getLabel(int n) {
        return null;
    }

    @Override
    public List<List<ClassPrediction>> decodePredictions(INDArray predictions, int n) {
        return null;
    }
}
