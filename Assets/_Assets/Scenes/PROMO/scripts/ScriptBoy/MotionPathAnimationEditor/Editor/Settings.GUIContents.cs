using UnityEngine;

namespace ScriptBoy.MotionPathAnimEditor
{
    static partial class Settings
    {
        static class GUIContents
        {
            public readonly static GUIContent handleSize = new GUIContent("Size", "Set the size of the editor handles.");
            public readonly static GUIContent handleColorNormal = new GUIContent("Normal Handle Color");
            public readonly static GUIContent handleColorSelected = new GUIContent("Selected Handle Color");
            public readonly static GUIContent handleCapControl = new GUIContent("Control Handle Cap");
            public readonly static GUIContent handleCapTangent = new GUIContent("Tangent Handle Cap");
            public readonly static GUIContent localSnappingIn2D = new GUIContent("Local Snapping In 2D", "By default you can only snap a handle to the world grid. You can enable this feature to snap a handle to the local grid related to the parent object.\n(It only works in 2D!)");

            public readonly static GUIContent pathSpace = new GUIContent("Space", "Local: The path shows the position of the object through local space.\n\nWorld: The path shows the position of the object through world space.");
            public readonly static GUIContent pathColorMode = new GUIContent("ColorMode", "Color: Draw paths with a single color.\n\nGradient: Draw paths with a gradient.\n(The path color changes based on the object’s velocity.)");
            public readonly static GUIContent pathAccuracy = new GUIContent("Accuracy", "Set the number of path segments between 2 keyframes.");
            public readonly static GUIContent pathColor = new GUIContent("Color");
            public const string pathGradient = "Gradient";
            public readonly static GUIContent pathFullName = new GUIContent("Show Full Name", "Show the motion path full name in the MotionPath list.");
            public readonly static GUIContent pathEditButton = new GUIContent("Show EditPath Button", "Show the EditPath button in the MotionPath list.\nIf you turn this off, all paths will be editable only based on the EditMode button.");

            public readonly static GUIContent applyRootOffset = new GUIContent("Apply Root Offset", "You can manually apply a custom offset to the path of the root object.");
            


            public readonly static GUIContent handleSizeSettings = new GUIContent("Handle Size", handleSize.tooltip);
            public readonly static GUIContent pathSpaceSettings = new GUIContent("Path Space", pathSpace.tooltip);
            public readonly static GUIContent pathColorModeSettings = new GUIContent("Path Color Mode", pathColorMode.tooltip);

        }
    }
}