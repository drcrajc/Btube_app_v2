package com.example.bubbleintube_v

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.media.MediaMetadataRetriever
import android.net.Uri
import android.os.Bundle
import android.provider.MediaStore
import android.view.View
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.TensorProcessor
import org.tensorflow.lite.support.common.TensorProcessor.Builder
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.nio.ByteBuffer
import java.nio.ByteOrder

class MainActivity : AppCompatActivity() {

    private val CAMERA_PERMISSION_REQUEST_CODE = 1001
    private val CAMERA_REQUEST_CODE = 1002
    private val SELECT_VIDEO_REQUEST_CODE = 1003

    private lateinit var btnRecordVideo: Button
    private lateinit var btnSelectVideo: Button
    private lateinit var btnCloseApp: Button
    private lateinit var imageView: ImageView
    private lateinit var videoThumbnailImageView: ImageView
    private lateinit var tvPredictResult: TextView
    private lateinit var tvResultAnalysis: TextView

    private lateinit var labels: List<String>
    private lateinit var interpreter: Interpreter

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        btnRecordVideo = findViewById(R.id.btnRecordVideo)
        btnSelectVideo = findViewById(R.id.btnSelectVideo)
        btnCloseApp = findViewById(R.id.btnCloseApp)
        imageView = findViewById(R.id.imageView)
        videoThumbnailImageView = findViewById(R.id.videoThumbnailImageView)
        tvPredictResult = findViewById(R.id.tvPredictResult)
        tvResultAnalysis = findViewById(R.id.tvResultAnalysis)

        // Load labels
        labels = loadLabelsFromAssets()

        // Load model
        interpreter = Interpreter(loadModelFile())

        btnRecordVideo.setOnClickListener {
            if (checkCameraPermission()) {
                recordVideo()
            } else {
                requestCameraPermission()
            }
        }

        btnSelectVideo.setOnClickListener {
            val intent = Intent(Intent.ACTION_GET_CONTENT)
            intent.type = "video/*"
            startActivityForResult(intent, SELECT_VIDEO_REQUEST_CODE)
        }

        btnCloseApp.setOnClickListener {
            finish()
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        interpreter.close()
    }

    private fun checkCameraPermission(): Boolean {
        val result = ContextCompat.checkSelfPermission(
            this,
            Manifest.permission.CAMERA
        )
        return result == PackageManager.PERMISSION_GRANTED
    }

    private fun requestCameraPermission() {
        ActivityCompat.requestPermissions(
            this,
            arrayOf(Manifest.permission.CAMERA),
            CAMERA_PERMISSION_REQUEST_CODE
        )
    }

    private fun recordVideo() {
        val intent = Intent(MediaStore.ACTION_VIDEO_CAPTURE)
        startActivityForResult(intent, CAMERA_REQUEST_CODE)
    }

    private fun loadLabelsFromAssets(): List<String> {
        // Load labels from assets folder
        val labels = assets.open("labels.txt").bufferedReader().use { it.readText() }
        return labels.split("\n")
    }

    private fun loadModelFile(): ByteBuffer {
        val assetFileDescriptor = assets.openFd("may3btxcp.tflite")
        val inputStream = assetFileDescriptor.createInputStream()
        val modelFileBuffer = inputStream.readBytes()
        val byteBuffer = ByteBuffer.allocateDirect(modelFileBuffer.size)
        byteBuffer.order(ByteOrder.nativeOrder())
        byteBuffer.put(modelFileBuffer)
        byteBuffer.flip()
        return byteBuffer
    }

    private fun classifyVideoFrames(videoUri: Uri) {
        val retriever = MediaMetadataRetriever()
        retriever.setDataSource(this, videoUri)
        val duration = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION)?.toLong()

        var positiveFrames = 0
        var negativeFrames = 0

        // Set video thumbnail
        val frame = retriever.getFrameAtTime()
        videoThumbnailImageView.setImageBitmap(frame)

        // Process video frames
        duration?.let {
            val framesPerSecond = 1
            val frameInterval = (1000 / framesPerSecond).toLong()
            val totalFramesCount = (duration / frameInterval) + 1

            for (time in 0..duration step frameInterval) {
                val frame = retriever.getFrameAtTime(time * 1000, MediaMetadataRetriever.OPTION_CLOSEST)
                val bitmap = frame?.let { Bitmap.createBitmap(it) }
                if (bitmap != null) {
                    val predictedLabel = classifyFrame(bitmap)
                    if (predictedLabel == "Positive") {
                        positiveFrames++
                    } else {
                        negativeFrames++
                    }
                }
            }

            // Calculate percentages
            val totalFrames = positiveFrames + negativeFrames
            val positivePercentage = (positiveFrames.toFloat() / totalFrames) * 100
            val negativePercentage = (negativeFrames.toFloat() / totalFrames) * 100

            // Update UI
            tvResultAnalysis.text = "Result analysis:\n" +
                    "Video length (in Sec): ${duration / 1000} Sec\n" +
                    "Video Frames (${framesPerSecond} per Sec): ${totalFramesCount}\n\n" +
                    "Predicted Result:\n" +
                    "Positive Frames: $positiveFrames\n" +
                    "Negative Frames: $negativeFrames\n\n" +
                    "Hence, the predicted result is ${
                        if (negativePercentage > 99) "Negative (${negativePercentage.toInt()}%)"
                        else "Positive (${positivePercentage.toInt()}%)"
                    }"
        }

        retriever.release()
    }

    private fun classifyFrame(bitmap: Bitmap): String {
        val resizedBitmap = Bitmap.createScaledBitmap(bitmap, 224, 224, true)
        val inputBuffer = convertBitmapToByteBuffer(resizedBitmap)

        // Run the model
        val outputBuffer = TensorBuffer.createFixedSize(intArrayOf(1, labels.size), DataType.FLOAT32)
        interpreter.run(inputBuffer, outputBuffer.buffer.rewind())

        // Get the predictions
        val outputArray = outputBuffer.floatArray
        val maxIndex = outputArray.indices.maxByOrNull { outputArray[it] } ?: -1

        // Display the predicted result
        val predictedLabel = labels[maxIndex]

        return predictedLabel
    }

    private fun convertBitmapToByteBuffer(bitmap: Bitmap): ByteBuffer {
        val byteBuffer = ByteBuffer.allocateDirect(4 * 224 * 224 * 3)
        byteBuffer.order(ByteOrder.nativeOrder())
        val intValues = IntArray(224 * 224)

        bitmap.getPixels(intValues, 0, bitmap.width, 0, 0, bitmap.width, bitmap.height)

        var pixel = 0
        for (i in 0 until 224) {
            for (j in 0 until 224) {
                val value = intValues[pixel++]
                byteBuffer.putFloat(((value shr 16) and 0xFF) / 255f)
                byteBuffer.putFloat(((value shr 8) and 0xFF) / 255f)
                byteBuffer.putFloat((value and 0xFF) / 255f)
            }
        }
        return byteBuffer
    }

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<out String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        when (requestCode) {
            CAMERA_PERMISSION_REQUEST_CODE -> {
                if (grantResults.isNotEmpty() && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                    recordVideo()
                }
            }
        }
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (resultCode == RESULT_OK) {
            when (requestCode) {
                CAMERA_REQUEST_CODE -> {
                    val videoUri = data?.data
                    if (videoUri != null) {
                        classifyVideoFrames(videoUri)
                    }
                }
                SELECT_VIDEO_REQUEST_CODE -> {
                    val videoUri = data?.data
                    if (videoUri != null) {
                        classifyVideoFrames(videoUri)
                    }
                }
            }
        }
    }
}
