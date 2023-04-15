using UnityEditor;
using UnityEngine;

namespace ScriptBoy.MotionPathAnimEditor
{
    static class ActiveAnimationWindow
    {
        public static AnimationWindow s_Instance;
        public static AnimationWindow instance
        {
            get
            {
                if (s_Instance == null)
                {
                    s_Instance = AnimationWindowUtility.GetOpenAnimationWindow();
                }

                return s_Instance;
            }
        }
        public static AnimationClip animationClip => instance.animationClip;
        public static GameObject root => instance.GetActiveRootGameObject();
    }
}