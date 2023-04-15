using UnityEngine;

namespace ScriptBoy.MotionPathAnimEditor
{
    static partial class Settings
    {
        static class Defaults
        {
            public const HandleCapShape HandleCapControl = HandleCapShape.Cube;
            public const HandleCapShape HandleCapTangent = HandleCapShape.Sphere;
            public static readonly Color HandleColorNormal = Color.white;
            public static readonly Color HandleColorSelected = Color.yellow;
            public const float HandleSize = 0.1f;


            public const ColorMode PathColorMode = ColorMode.Color;
            public const Space PathSpace = Space.Local;
            public static readonly Color PathColor;
            public static readonly MinMaxGradient PathGradient = new MinMaxGradient("#009EFF", "#FF0000");
            public const int PathAccuracy = 30;
            public const bool PathFullName = false;
            public const bool PathEditButton = true;

            public static readonly Color WindowColor;

            public const bool LocalSnappingIn2D = false;
            public const bool ApplyRootOffset = false; 

            static Defaults()
            {
                ColorUtility.TryParseHtmlString("#00C0FF", out PathColor);
                ColorUtility.TryParseHtmlString("#0079FF", out WindowColor);
            }
        }
    }
}