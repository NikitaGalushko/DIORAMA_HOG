using UnityEditor;
using UnityEngine;

namespace ScriptBoy.MotionPathAnimEditor
{
    partial class AnimEditorWindow : EditorWindow
    {
        private static AnimEditorWindow s_Instance;

        [SerializeField] private Material m_CurveMaterial;

        private AnimEditor m_AnimEditor;
        private HandleSelectionTransformEditor m_SelectionTransformEditor;
        private RootOffsetEditor m_RootOffsetEditor;
        private MotionPathListRenderer m_ListRenderer;

        private bool m_ShowFullName;
        private bool m_ShowSettings;
        private bool m_EditMode;

        [MenuItem("Tools/ScriptBoy/Motion Path Animation Editor", false, 0)]
        static void OpenWindow()
        {
            GetWindow<AnimEditorWindow>().Show();
        }

        public static void RepaintWindow()
        {
            if (s_Instance != null) s_Instance.Repaint();
        }

        private void Awake()
        {
            titleContent = new GUIContent("Motion Path Anim Editor");
            minSize = new Vector2(300, 200);
        }

        private void OnEnable()
        {
            s_Instance = this;
            BezierCurveRenderer.SetMaterial(m_CurveMaterial);
            m_SelectionTransformEditor = (HandleSelectionTransformEditor)Editor.CreateEditor(HandleSelectionTransform.instance);
            m_RootOffsetEditor = (RootOffsetEditor)Editor.CreateEditor(RootOffset.instance);
            m_AnimEditor = new AnimEditor();
            m_ListRenderer = new MotionPathListRenderer(m_AnimEditor.motionPaths);
            m_AnimEditor.editMode = m_EditMode;
        }

        private void OnDestroy()
        {
            DestroyImmediate(m_SelectionTransformEditor);
            m_AnimEditor.Destroy();
            Tools.hidden = false;
        }


        private void OnGUI()
        {
            //bool wideMode = EditorGUIUtility.wideMode;
            //EditorGUIUtility.wideMode = true;

            if (OnCheckStateGUI())
            {
                Tools.hidden = false;
                return;
            }

            Tools.hidden = m_EditMode;

            OnHeaderGUI();
            OnBodyGUI();

            //EditorGUIUtility.wideMode = wideMode;
        }

        private bool OnCheckStateGUI()
        {
            if (AnimEditor.animationWindow == null)
            {
                AnimEditor.animationWindow = AnimationWindowUtility.GetOpenAnimationWindow();
            }

            if (AnimEditor.animationWindow == null)
            {
                EditorGUILayout.HelpBox("No Animation Window!", MessageType.Error);
                if (GUILayout.Button("Open Animation Window"))
                {
                    AnimEditor.animationWindow = AnimationWindowUtility.OpenAnimationWindow();
                }
                return true;
            }

            if (AnimEditor.animationWindow.animationClip == null)
            {
                if (AnimEditor.animationWindow.hasFocus)
                {
                    EditorGUILayout.HelpBox("No Animation Clip!", MessageType.Error);
                }
                else
                {
                    EditorGUILayout.HelpBox("No Animation Window!", MessageType.Warning);
                    if (GUILayout.Button("Open Animation Window"))
                    {
                        AnimEditor.animationWindow.Focus();
                    }
                }
                return true;
            }

            if (Selection.activeGameObject == null || AnimEditor.animationWindow.GetActiveRootGameObject() == null)
            {
                EditorGUILayout.HelpBox("No Root GameObject!", MessageType.Error);
                return true;
            }

            return false;
        }

        private void OnHeaderGUI()
        {
            using (new GUILayout.HorizontalScope(GUIStyles.header))
            {
                m_EditMode = GUILayout.Toggle(m_EditMode, GUIContents.editMode, GUIStyles.toggleEditMode);
                GUILayout.FlexibleSpace();
                m_ShowSettings = GUILayout.Toggle(m_ShowSettings, GUIContents.showSettings, GUIStyles.toggleSettings);

                m_AnimEditor.editMode = m_EditMode;
            }
        }

        private void OnBodyGUI()
        {
            if (m_ShowSettings)
            {
                Settings.OnFoldoutWindowGUI();
            }

            if (Settings.applyRootOffset)
            {
                OnRootOffsetGUI();
            }

            if (m_EditMode)
            {
                if (HandleSelection.count == 1)
                {
                    OnHandleGUI();
                }
                else if (HandleSelection.count > 1)
                {
                    OnSelectionTransformGUI();
                }
            }

            m_ListRenderer.DoLayoutList();
        }

        private void OnRootOffsetGUI()
        {
            EditorGUI.BeginChangeCheck();
            m_RootOffsetEditor.OnGUI();
            if (EditorGUI.EndChangeCheck())
            {
                SceneView.RepaintAll();
            }
        }

        private void OnSelectionTransformGUI()
        {
            EditorGUI.BeginChangeCheck();
            m_SelectionTransformEditor.OnGUI();
            if (EditorGUI.EndChangeCheck())
            {
                HandleSelectionTransform.instance.UpdatePositions();
                m_AnimEditor.ApplyChages();
            }
        }

        private void OnHandleGUI()
        {
            EditorGUI.BeginChangeCheck();
            using (new CustomGUILayout.FoldoutWindowScope("Handle", out bool open))
            {
                if (open)
                {
                    HandleSelection.activeHandle.position = EditorGUILayout.Vector3Field("Position", HandleSelection.activeHandle.position);
                }
            }
            if (EditorGUI.EndChangeCheck())
            {
                HandleSelection.activeHandle.hasChanged = true;
                m_AnimEditor.ApplyChages();
            }
        }
    }
}