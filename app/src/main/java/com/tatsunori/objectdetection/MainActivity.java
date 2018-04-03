package com.tatsunori.objectdetection;

import android.Manifest;
import android.annotation.SuppressLint;
import android.app.Application;
import android.content.Context;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.Rect;
import android.graphics.RectF;
import android.net.Uri;
import android.os.AsyncTask;
import android.provider.MediaStore;
import android.support.annotation.NonNull;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.view.Menu;
import android.view.MenuInflater;
import android.view.MenuItem;
import android.view.View;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import com.qualcomm.qti.snpe.NeuralNetwork;
import com.qualcomm.qti.snpe.SNPE;

import java.io.File;
import java.io.FileFilter;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.List;

import butterknife.BindView;
import butterknife.ButterKnife;

public class MainActivity extends AppCompatActivity {

    private final int REQUEST_CODE_PICKER = 1;

    private Bitmap mBitmap;

    private DetectionTask mDteDetectionTask;

    private NeuralNetwork.Runtime mTargetRuntime;

    private String mModelPath;

    @BindView(R.id.imageView)
    ImageView mImageView;

    @BindView(R.id.textView)
    TextView mTextView;

    enum MenuRuntimeGroup {

        SelectCpuRuntime(NeuralNetwork.Runtime.CPU),
        SelectGpuRuntime(NeuralNetwork.Runtime.GPU),
        SelectDspRuntime(NeuralNetwork.Runtime.DSP);

        public static int ID = 1;

        public NeuralNetwork.Runtime runtime;

        MenuRuntimeGroup(NeuralNetwork.Runtime runtime) {
            this.runtime = runtime;
        }
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        ButterKnife.bind(this);

        Log.d("@@@", "SNPE Version:" + SNPE.getRuntimeVersion(getApplication()));
        boolean isGPU = new SNPE.NeuralNetworkBuilder(getApplication()).isRuntimeSupported(NeuralNetwork.Runtime.GPU);
        boolean isDSP = new SNPE.NeuralNetworkBuilder(getApplication()).isRuntimeSupported(NeuralNetwork.Runtime.DSP);
        Log.d("@@@", "GPU:" + isGPU + " DSP:" + isDSP);

        copyToLocal("mobilenet_ssd.dlc");
        File file = getFilesDir();
        mModelPath = file.getAbsolutePath() + "/mobilenet_ssd.dlc";
        mTargetRuntime = NeuralNetwork.Runtime.CPU;

        mDteDetectionTask = new DetectionTask(this);
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);

        if (REQUEST_CODE_PICKER == requestCode && data != null) {
            try {
                mBitmap = MediaStore.Images.Media.getBitmap(getContentResolver(), data.getData());
            } catch (IOException e) {
                e.printStackTrace();
            }
            showImage(mBitmap);
            mDteDetectionTask.execute(mBitmap);
        }
    }

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        MenuInflater inflater = getMenuInflater();
        inflater.inflate(R.menu.menu, menu);

        final SNPE.NeuralNetworkBuilder builder = new SNPE.NeuralNetworkBuilder(
                (Application) (getApplicationContext()));
        for (MenuRuntimeGroup item : MenuRuntimeGroup.values()) {
            if (builder.isRuntimeSupported(item.runtime)) {
                menu.add(MenuRuntimeGroup.ID, item.ordinal(), 0, item.runtime.name());
            }
        }
        return super.onCreateOptionsMenu(menu);
    }

    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        if (item.getGroupId() == MenuRuntimeGroup.ID) {
            final MenuRuntimeGroup option = MenuRuntimeGroup.values()[item.getItemId()];
            mTargetRuntime = option.runtime;
            return true;
        }
        switch (item.getItemId()) {
            case R.id.file_open:
                startImagePicker();
                return true;
            default:
                break;
        }
        return super.onOptionsItemSelected(item);
    }

    private void startImagePicker() {
        Intent photoPickerIntent = new Intent(Intent.ACTION_PICK);
        photoPickerIntent.setType("image/*");
        startActivityForResult(photoPickerIntent, REQUEST_CODE_PICKER);
    }

    private void showImage(Bitmap bmp) {
        mImageView.setImageBitmap(bmp);
        mTextView.setVisibility(View.INVISIBLE);
    }

    private Bitmap drawRect(List<DetectionInfo> list) {
        Bitmap bitmap = Bitmap.createBitmap(mBitmap.getWidth(), mBitmap.getHeight(),
                Bitmap.Config.ARGB_8888);
        Canvas canvas = new Canvas(bitmap);
        Paint paint = new Paint();
        paint.setFilterBitmap(true);
        paint.setColor(Color.RED);
        paint.setStyle(Paint.Style.STROKE);
        paint.setStrokeWidth(10.0f);
        paint.setTextSize(100.0f);

        canvas.drawBitmap(mBitmap, 0, 0, paint);

        for (DetectionInfo info : list) {
            RectF rect = info.getRect();
            canvas.drawText(info.getClassName(), rect.left, rect.top, paint);
            canvas.drawRect(info.getRect(), paint);
        }

        return bitmap;
    }

    private void copyToLocal(String fileName) {
        Log.d("@@@", "copyToLocal() start");
        try {
            InputStream inputStream = getAssets().open(fileName);
            FileOutputStream fileOutputStream = openFileOutput(fileName, MODE_PRIVATE);
            byte[] buffer = new byte[1024];
            int length = 0;
            while ((length = inputStream.read(buffer)) >= 0) {
                fileOutputStream.write(buffer, 0, length);
            }
            fileOutputStream.close();
            inputStream.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
        Log.d("@@@", "copyToLocal() end");
    }

    class DetectionTask extends AsyncTask<Bitmap, Bitmap, Bitmap> {
        private TinyYolo mTinyYolo;
        private SnpeMobileNetSsd mSsd;

        DetectionTask(Context context) {
            //mTinyYolo = new TinyYolo(context);
            File model = new File(mModelPath);
            mSsd = new SnpeMobileNetSsd(getApplication(), mTargetRuntime, model);
        }

        @Override
        protected Bitmap doInBackground(Bitmap... bitmaps) {
//            return drawRect(mTinyYolo.detection(bitmaps[0]));
            return drawRect(mSsd.detection(bitmaps[0]));
        }

        @Override
        protected void onPostExecute(Bitmap bitmap) {
            mImageView.setImageBitmap(bitmap);
        }
    }
}
