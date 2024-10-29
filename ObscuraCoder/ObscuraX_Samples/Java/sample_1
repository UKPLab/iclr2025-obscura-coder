package com.dji.GSDemo.GoogleMap;

import android.os.Bundle;
import android.support.v7.app.AppCompatActivity;
import android.content.Intent;
import android.widget.TextView;

/**
 * Created by lasithniro on 1/4/17.
**/

public class showDetail extends AppCompatActivity {
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.show_details);

        TextView result = (TextView) findViewById(R.id.textView2);
        TextView resultData = (TextView) findViewById(R.id.textView3);
        Intent in = getIntent();
        result.setText(in.getStringExtra("name"));
        resultData.setText(in.getStringExtra("data"));
    }
}