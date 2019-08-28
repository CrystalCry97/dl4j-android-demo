package com.yptheangel.dl4jandroid.yolo_objdetection;


import android.Manifest;
import android.content.Context;
import android.content.Intent;
import android.content.pm.PackageManager;

import android.os.Build;
import android.support.v4.app.ActivityCompat;
import android.support.v4.content.ContextCompat;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.Toast;


public class MainActivity extends AppCompatActivity {
    public static final int REQUEST_PERMISSION = 300;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        // explictly request permission from user to use the phone resource
        if (ActivityCompat.checkSelfPermission(this.getApplicationContext(), android.Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED){
            ActivityCompat.requestPermissions(this, new String[] {android.Manifest.permission.CAMERA}, REQUEST_PERMISSION);
        }

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M
                && ContextCompat.checkSelfPermission(this, Manifest.permission.WRITE_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.WRITE_EXTERNAL_STORAGE},
                    REQUEST_PERMISSION);
        }

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M
                && ContextCompat.checkSelfPermission(this, Manifest.permission.READ_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.READ_EXTERNAL_STORAGE},
                    REQUEST_PERMISSION);
        }

            findViewById(R.id.start_butt).setOnClickListener(new View.OnClickListener() {

                @Override
            public void onClick(View view) {
//                Context context = getApplicationContext();
                Toast.makeText(getApplicationContext(), "Real Time Inference started.", Toast.LENGTH_LONG).show();
                startActivity(new Intent(MainActivity.this, ObjDetection.class));
            }
        });


            findViewById(R.id.singlestart_butt).setOnClickListener(new View.OnClickListener() {
            @Override
                public void onClick(View view){
                    Toast.makeText(getApplicationContext(), "Single Image Inference started.", Toast.LENGTH_LONG).show();
                    startActivity(new Intent(MainActivity.this, PredictSingle.class));
                }
            });




    }

}
