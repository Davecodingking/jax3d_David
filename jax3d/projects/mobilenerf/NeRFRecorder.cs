using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Text;
using UnityEngine;

public class NeRFRecorder : MonoBehaviour
{
    [Header("文件保存设置")]
    public string folderName = "Data_Sponza_1";
    public int targetFrameCount = 100;
    public float captureInterval = 0.1f;

    [Header("数据集划分")]
    public bool autoSplitData = true;
    public int testInterval = 8;

    [Header("轨迹参数")]
    public bool useAutoRotation = true;

    [Tooltip("X轴半径 (宽/短边)：建议设为 2.5")]
    public float widthRadius = 2.5f;

    [Tooltip("Z轴半径 (深/长边)：建议设为 6.5")]
    public float depthRadius = 6.5f;

    [Tooltip("相机高度")]
    public float height = 1.7f;

    [Tooltip("相机看向中心点的高度")]
    public float lookAtHeight = 1.0f;

    private string savePath;
    private Camera cam;
    private List<string> allFramesJson = new List<string>();
    private float timer = 0;
    private int capturedCount = 0;
    private bool isRecording = false;

    void Start()
    {
        cam = GetComponent<Camera>();
        savePath = Path.Combine(Application.dataPath, "..", folderName);

        string imagesPath = Path.Combine(savePath, "train");
        if (!Directory.Exists(imagesPath)) Directory.CreateDirectory(imagesPath);

        Debug.Log($"📂 数据保存路径: {savePath}");
        isRecording = true;
    }

    void Update()
    {
        if (!isRecording) return;

        if (useAutoRotation)
        {
            float progress = (float)capturedCount / targetFrameCount;
            float angle = progress * 2 * Mathf.PI;

            // 修正后的半径逻辑：Width 对应 X，Depth 对应 Z
            float x = Mathf.Sin(angle) * widthRadius;
            float z = Mathf.Cos(angle) * depthRadius;

            transform.position = new Vector3(x, height, z);
            transform.LookAt(new Vector3(0, lookAtHeight, 0));
        }

        timer += Time.deltaTime;
        if (timer >= captureInterval)
        {
            timer = 0;
            CaptureFrame();
        }
    }

    /// <summary>
    /// 核心逻辑：输出相机坐标 X 轴翻转后的矩阵
    /// 策略：仅翻转 Right 向量。
    /// </summary>
    void CaptureFrame()
    {
        string filename = $"r_{capturedCount}";
        string fullPath = Path.Combine(savePath, "train", filename + ".png");

        ScreenCapture.CaptureScreenshot(fullPath);

        // 1. 获取 Unity 世界空间下的原始位姿
        Vector3 p = transform.position;
        Vector3 r = transform.right;   // Unity +X (Right)
        Vector3 u = transform.up;      // Unity +Y (Up)
        Vector3 f = transform.forward; // Unity +Z (Forward)

        // 2. 构建相机到世界矩阵 (C2W)
        // Col0 (X) = -Right
        // Col1 (Y) = Up
        // Col2 (Z) = Forward
        // Col3 (Pos) = Position

        Vector3 col0 = -r;         // Right
        Vector3 col1 = u;          // Up
        Vector3 col2 = f;          // Forward
        Vector3 col3 = p;          // Position

        // 3. 写入 JSON (Row-Major 格式)
        StringBuilder sb = new StringBuilder();
        sb.Append("    {\n");
        sb.Append($"      \"file_path\": \"./train/{filename}\",\n");
        sb.Append("      \"transform_matrix\": [\n");
        // Row 0
        sb.Append($"        [{col0.x:F6}, {col1.x:F6}, {col2.x:F6}, {col3.x:F6}],\n");
        // Row 1
        sb.Append($"        [{col0.y:F6}, {col1.y:F6}, {col2.y:F6}, {col3.y:F6}],\n");
        // Row 2
        sb.Append($"        [{col0.z:F6}, {col1.z:F6}, {col2.z:F6}, {col3.z:F6}],\n");
        // Row 3
        sb.Append("        [0.000000, 0.000000, 0.000000, 1.000000]\n");
        sb.Append("      ]\n");
        sb.Append("    }");

        allFramesJson.Add(sb.ToString());
        capturedCount++;

        if (capturedCount >= targetFrameCount)
        {
            SaveAllJsons();
            isRecording = false;
            Debug.Log("✅ 数据录制完成！已切换为无镜像对齐协议。");
#if UNITY_EDITOR
            UnityEditor.EditorApplication.isPlaying = false;
#endif
        }
    }

    void SaveAllJsons()
    {
        float vFovRad = cam.fieldOfView * Mathf.Deg2Rad;
        float hFovRad = 2 * Mathf.Atan(Mathf.Tan(vFovRad / 2) * cam.aspect);
        string cameraAngleStr = hFovRad.ToString("F6");

        if (!autoSplitData)
        {
            WriteJsonFile("transforms_train.json", cameraAngleStr, allFramesJson);
        }
        else
        {
            List<string> trainFrames = new List<string>();
            List<string> valFrames = new List<string>();
            List<string> testFrames = new List<string>();

            for (int i = 0; i < allFramesJson.Count; i++)
            {
                if (i % testInterval == 0)
                {
                    valFrames.Add(allFramesJson[i]);
                    testFrames.Add(allFramesJson[i]);
                }
                else
                {
                    trainFrames.Add(allFramesJson[i]);
                }
            }

            WriteJsonFile("transforms_train.json", cameraAngleStr, trainFrames);
            WriteJsonFile("transforms_val.json", cameraAngleStr, valFrames);
            WriteJsonFile("transforms_test.json", cameraAngleStr, testFrames);
        }
    }

    void WriteJsonFile(string filename, string angleStr, List<string> frames)
    {
        StringBuilder finalJson = new StringBuilder();
        finalJson.Append("{\n");
        finalJson.Append($"  \"camera_angle_x\": {angleStr},\n");
        finalJson.Append("  \"frames\": [\n");
        for (int i = 0; i < frames.Count; i++)
        {
            finalJson.Append(frames[i]);
            if (i < frames.Count - 1) finalJson.Append(",\n");
        }
        finalJson.Append("\n  ]\n}");
        string jsonPath = Path.Combine(savePath, filename);
        File.WriteAllText(jsonPath, finalJson.ToString());
    }
}
