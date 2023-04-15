using UnityEngine;

namespace ScriptBoy.MotionPathAnimEditor
{
    static class Textures
    {
        private static Color s_Color;
        private static Color[] s_Colors;

        private const int kWidth = 100;
        private const int kHeight = 1;

        public static Texture2D windowHeader { get; }
        public static Texture2D foldoutWindowTab { get; }
        public static Texture2D itemRowNormal { get; }
        public static Texture2D itemRowActive { get; }

        static Textures()
        {
            s_Colors = new Color[kWidth];

            windowHeader = CreateTexture();
            foldoutWindowTab = CreateTexture();
            itemRowNormal = CreateTexture();
            itemRowActive = CreateTexture();

            RefreshColor(Settings.windowColor);
        }

        public static void RefreshColor(Color color)
        {
            if (s_Color != color)
            {
                s_Color = color;
                RefreshTextures();
                AnimEditorWindow.RepaintWindow();
            }
        }

        private static Texture2D CreateTexture()
        {
            var obj = new Texture2D(kWidth, kHeight);
            obj.hideFlags = HideFlags.DontSave;
            return obj;
        }

        private static void RefreshTextures()
        {
            RefreshTexture(windowHeader, 0.8f, 1.0f);
            RefreshTexture(foldoutWindowTab, 0.5f, 1.0f);
            RefreshTexture(itemRowNormal, 0.25f, 0.75f);
            RefreshTexture(itemRowActive, 0.5f, 1.0f);
        }

        private static void RefreshTexture(Texture2D texture, float vMin,float vMax)
        {
            ColorHSV colorHSV = new ColorHSV(s_Color);
            float iMax = kWidth - 1;
            for (int i = 0; i < kWidth; i++)
            {
                float t = i / iMax;
                colorHSV.v = Mathf.Lerp(vMin, vMax, t);
                s_Colors[i] = colorHSV.ToRGB;
            }
            texture.SetPixels(s_Colors);
            texture.Apply();
        }
    }
}