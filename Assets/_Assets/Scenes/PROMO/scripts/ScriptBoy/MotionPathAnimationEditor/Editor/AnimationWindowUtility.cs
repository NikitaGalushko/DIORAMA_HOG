using UnityEditor;
using UnityEngine;

namespace ScriptBoy.MotionPathAnimEditor
{
    internal static class AnimationWindowUtility
    {
        public static AnimationWindow OpenAnimationWindow()
        {

            return EditorWindow.GetWindow<AnimationWindow>();
        }

        public static AnimationWindow GetOpenAnimationWindow()
        {
            var windows = Resources.FindObjectsOfTypeAll<AnimationWindow>();

            if (windows.Length > 0)
            {
                return windows[0];
            }

            return null;
        }

        public static GameObject GetActiveRootGameObject(this AnimationWindow animationWindow)
        {
            GameObject go = Selection.activeGameObject;
            if (go == null) return null;
            Animator animator = go.GetComponentInParent<Animator>();
            if (animator == null) return null;
            return animator.gameObject;
        }
    }
}


/***************************
 * UnityEditor Source Code *
 ***************************


namespace UnityEditor
{
    internal class AnimationWindow : EditorWindow
    {
        private AnimEditor m_AnimEditor;
    }

    internal class AnimEditor : ScriptableObject
    {
        [SerializeField]
        private AnimationWindowState m_State;
        [SerializeField]
		private CurveEditor m_CurveEditor;
    }

    [Serializable]
	internal class CurveEditor : TimeArea, CurveUpdater
	{

    	internal List<CurveSelection> selectedCurves
		{
			get
			{
				return this.selection.selectedCurves;
			}
			set
			{
				this.selection.selectedCurves = value;
				this.InvalidateSelectionBounds();
			}
		}

    	internal CurveWrapper GetCurveWrapperFromSelection(CurveSelection curveSelection)
		{
			return this.GetCurveWrapperFromID(curveSelection.curveID);
		}
    }

    internal class CurveWrapper
	{
    	public AnimationCurve curve
		{
			get
			{
				return this.renderer.GetCurve();
			}
		}

        public EditorCurveBinding binding;
    }
}


namespace UnityEditorInternal
{
    internal class AnimationWindowState : ScriptableObject
    {
        public AnimationClip activeAnimationClip
        {
            get
            {
                return this.selection.animationClip;
            }
            set
            {
                bool canChangeAnimationClip = this.selection.canChangeAnimationClip;
                if (canChangeAnimationClip)
                {
                    this.selection.animationClip = value;
                    this.OnSelectionChanged();
                }
            }
        }

        public GameObject activeRootGameObject
        {
            get
            {
                return this.selection.rootGameObject;
            }
        }

        public int currentFrame
        {
            get
            {
                return this.time.frame;
            }
            set
            {
                this.controlInterface.GoToFrame(value);
            }
        }
    }
}
*/
