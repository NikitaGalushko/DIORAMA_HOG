using UnityEditor;
using UnityEngine;

#if SETTINGS_MANAGER
using UnityEditor.SettingsManagement;
#endif

namespace ScriptBoy.MotionPathAnimEditor
{
    static partial class Settings
    {
#if SETTINGS_MANAGER
        private static UnityEditor.SettingsManagement.Settings s_SettingsInstance;

        private static UnityEditor.SettingsManagement.Settings instance
        {
            get
            {
                if (s_SettingsInstance == null)
                    s_SettingsInstance = new UnityEditor.SettingsManagement.Settings("ScriptBoy.MotionPathAnimEditor");
                return s_SettingsInstance;
            }
        }
#endif

#region Variables
        private static UserSetting<float> s_HandleSize = new UserSetting<float>(Keys.HandleSize, Defaults.HandleSize);
        private static UserSetting<HandleCapShape> s_HandleCapControl = new UserSetting<HandleCapShape>(Keys.HandleCapControl, Defaults.HandleCapControl);
        private static UserSetting<HandleCapShape> s_HandleCapTangent = new UserSetting<HandleCapShape>(Keys.HandleCapTangent, Defaults.HandleCapTangent);
        private static UserSetting<Color> s_HandleColorNormal = new UserSetting<Color>(Keys.HandleColorNormal, Defaults.HandleColorNormal);
        private static UserSetting<Color> s_HandleColorSelected = new UserSetting<Color>(Keys.HandleColorSelected, Defaults.HandleColorSelected);
        private static UserSetting<bool> s_LocalSnappingIn2D = new UserSetting<bool>(Keys.LocalSnappingIn2D, Defaults.LocalSnappingIn2D);

        private static UserSetting<int> s_PathAccuracy = new UserSetting<int>(Keys.PathAccuracy, Defaults.PathAccuracy);
        private static UserSetting<ColorMode> s_PathColorMode = new UserSetting<ColorMode>(Keys.PathColorMode, Defaults.PathColorMode);
        private static UserSetting<Color> s_PathColor = new UserSetting<Color>(Keys.PathColor, Defaults.PathColor);
        private static UserSetting<MinMaxGradient> s_PathGradient = new UserSetting<MinMaxGradient>(Keys.PathGradient, Defaults.PathGradient);
        private static UserSetting<Space> s_PathSpace = new UserSetting<Space>(Keys.PathSpace, Defaults.PathSpace);
        private static UserSetting<bool> s_PathFullName = new UserSetting<bool>(Keys.PathFullName, Defaults.PathFullName);
        private static UserSetting<bool> s_PathEditButton= new UserSetting<bool>(Keys.PathEditButton, Defaults.PathEditButton);

        private static UserSetting<bool> s_ApplyRootOffset = new UserSetting<bool>(Keys.ApplyRootOffset, Defaults.ApplyRootOffset);

        private static UserSetting<Color> s_WindowColor = new UserSetting<Color>(Keys.WindowColor, Defaults.WindowColor);
#endregion

#region Properties
        public static float handleSize
        {
            get => s_HandleSize.Value;
            private set => s_HandleSize.Value = value;
        }

        public static Color handleColorNormal
        {
            get => s_HandleColorNormal.Value;
            private set => s_HandleColorNormal.Value = value;
        }

        public static Color handleColorSelected
        {
            get => s_HandleColorSelected.Value;
            private set => s_HandleColorSelected.Value = value;
        }

        public static HandleCapShape handleCapControl
        {
            get => s_HandleCapControl.Value;
            private set => s_HandleCapControl.Value = value;
        }

        public static HandleCapShape handleCapTangent
        {
            get => s_HandleCapTangent.Value;
            private set => s_HandleCapTangent.Value = value;
        }

        public static bool localSnappingIn2D
        {
            get => s_LocalSnappingIn2D.Value;
            private set => s_LocalSnappingIn2D.Value = value;
        }


        public static ColorMode pathColorMode
        {
            get => s_PathColorMode.Value;
            private set => s_PathColorMode.Value = value;
        }

        public static Space pathSpace
        {
            get => s_PathSpace.Value;
            private set => s_PathSpace.Value = value;
        }

        public static Color pathColor
        {
            get => s_PathColor.Value;
            private set => s_PathColor.Value = value;
        }

        public static MinMaxGradient pathGradient
        {
            get => s_PathGradient.Value;
            private set => s_PathGradient.Value = value;
        }

        public static int pathAccuracy
        {
            get => s_PathAccuracy.Value;
            private set => s_PathAccuracy.Value = value;
        }

        public static bool pathFullName
        {
            get => s_PathFullName.Value;
            private set => s_PathFullName.Value = value;
        }

        public static bool pathEditButton
        {
            get => s_PathEditButton.Value;
            private set => s_PathEditButton.Value = value;
        }

        public static bool applyRootOffset
        {
            get => s_ApplyRootOffset.Value;
            private set => s_ApplyRootOffset.Value = value;
        }

        public static Color windowColor
        {
            get => s_WindowColor.Value;
            private set => s_WindowColor.Value = value;
        }
#endregion

#region Methods

#if SETTINGS_MANAGER
        [SettingsProvider]
        private static SettingsProvider CreateSettingsProvider()
        {
            var assembly = new[] { typeof(Settings).Assembly };
            var provider = new UserSettingsProvider("Preferences/Motion Path Animation Editor", instance, assembly);
            return provider;
        }

        [UserSettingBlock(" ")]
        private static void OnPreferencesGUI(string searchContext)
        {
            EditorGUI.BeginChangeCheck();

            OnHandleGUI();
            OnPathGUI();

            using (new CustomGUILayout.GroupScope("Window"))
            {
                windowColor = EditorGUILayout.ColorField("Color", windowColor);
            }

            if (EditorGUI.EndChangeCheck())
                ApplyChanges();

            if (GUILayout.Button("Reset to default settings"))
                Reset();
        }
#endif

        private static void OnHandleGUI()
        {
            using (new CustomGUILayout.GroupScope("Handle"))
            {
                handleSize = EditorGUILayout.Slider(GUIContents.handleSize, handleSize, 0.075f, 0.4f);
                handleColorNormal = EditorGUILayout.ColorField(GUIContents.handleColorNormal, handleColorNormal);
                handleColorSelected = EditorGUILayout.ColorField(GUIContents.handleColorSelected, handleColorSelected);
                handleCapControl = (HandleCapShape)EditorGUILayout.EnumPopup(GUIContents.handleCapControl, handleCapControl);
                handleCapTangent = (HandleCapShape)EditorGUILayout.EnumPopup(GUIContents.handleCapTangent, handleCapTangent);
                localSnappingIn2D = EditorGUILayout.Toggle(GUIContents.localSnappingIn2D, localSnappingIn2D);
            }
        }

        private static void OnPathGUI()
        {
            using (new CustomGUILayout.GroupScope("Path"))
            {
                pathSpace = (Space)EditorGUILayout.EnumPopup(GUIContents.pathSpace, pathSpace);
                if (pathSpace == Space.Local)
                {
                    pathColorMode = (ColorMode)EditorGUILayout.EnumPopup(GUIContents.pathColorMode, pathColorMode);
                    if (pathColorMode == ColorMode.Color)
                    {
                        pathColor = EditorGUILayout.ColorField(GUIContents.pathColor, pathColor);
                    }
                    else
                    {
                        pathGradient = CustomGUILayout.MinMaxGradientField(GUIContents.pathGradient, pathGradient);
                        pathAccuracy = EditorGUILayout.IntSlider(GUIContents.pathAccuracy, pathAccuracy, 0, 100);
                    }
                }
                else pathColor = EditorGUILayout.ColorField(GUIContents.pathColor, pathColor);

                pathFullName = EditorGUILayout.Toggle(GUIContents.pathFullName, pathFullName);
                pathEditButton = EditorGUILayout.Toggle(GUIContents.pathEditButton, pathEditButton);
                applyRootOffset = EditorGUILayout.Toggle(GUIContents.applyRootOffset, applyRootOffset);
            }
        }

        public static void OnFoldoutWindowGUI()
        {
            EditorGUI.BeginChangeCheck();
            using (new CustomGUILayout.FoldoutWindowScope("Settings", out bool open))
            {
                if (open)
                {
                    handleSize = EditorGUILayout.Slider(GUIContents.handleSizeSettings, handleSize, 0.075f, 0.4f);
                    pathSpace = (Space)EditorGUILayout.EnumPopup(GUIContents.pathSpaceSettings, pathSpace);
                    if (pathSpace == Space.Local)
                    {
                        pathColorMode = (ColorMode)EditorGUILayout.EnumPopup(GUIContents.pathColorModeSettings, pathColorMode);
                    }
                }
            }

            if (EditorGUI.EndChangeCheck())
                ApplyChanges();
        }

        private static void ApplyChanges()
        {
#if SETTINGS_MANAGER
            instance.Save();
#endif
            SceneView.RepaintAll();
            Textures.RefreshColor(windowColor);
        }

        private static void Reset()
        {
            s_HandleSize.Reset();
            s_HandleCapControl.Reset();
            s_HandleCapTangent.Reset();
            s_HandleColorNormal.Reset();
            s_HandleColorSelected.Reset();

            s_PathAccuracy.Reset();
            s_PathColorMode.Reset();
            s_PathColor.Reset();
            s_PathGradient.Reset();
            s_PathSpace.Reset();

            s_LocalSnappingIn2D.Reset();

            s_WindowColor.Reset();
            ApplyChanges();

        }
#endregion
    }
}