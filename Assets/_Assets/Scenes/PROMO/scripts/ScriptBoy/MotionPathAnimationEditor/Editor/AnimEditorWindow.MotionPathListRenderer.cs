using System.Collections.Generic;
using UnityEditor;
using UnityEditorInternal;
using UnityEngine;

namespace ScriptBoy.MotionPathAnimEditor
{
    partial class AnimEditorWindow
    {
        class MotionPathListRenderer : ReorderableList
        {
            private List<MotionPath> m_List;

            public MotionPathListRenderer(List<MotionPath> list) : base(list, typeof(MotionPath))
            {
                this.m_List = list;
                this.headerHeight = 5;
                this.elementHeight = 24;
                this.onAddCallback += OnAddCallback;
                this.onCanAddCallback += OnCanAddCallback;
                this.drawHeaderCallback += DrawHeaderCallback;
                this.drawElementCallback += DrawElementCallback;
                this.drawElementBackgroundCallback += DrawElementBackgroundCallback;
            }

            private void DrawElementBackgroundCallback(Rect rect, int index, bool isActive, bool isFocused)
            {
                GUI.DrawTexture(rect, isActive ? Textures.itemRowActive : Textures.itemRowNormal);
            }

            private void DrawElementCallback(Rect rect, int index, bool isActive, bool isFocused)
            {
                var motionPath = m_List[index];
                GUI.Label(rect, Settings.pathFullName ? motionPath.fullName : motionPath.name);

                float s = rect.height - 2;
                rect = new Rect(rect.x + rect.width - s * 3, rect.y, s, s);

                float m = 4;

                GUI.enabled = motionPath.active;
                if (Settings.pathEditButton)
                {
                    if (motionPath.HasCurveData)
                    {
                        motionPath.loop = GUI.Toggle(SetMargin(rect, m), motionPath.loop, GUIContents.loopPath, GUIStyles.toggleLoop);
                    }

                    rect.x += s;

                    if (motionPath.HasCurveData)
                    {
                        motionPath.editable = GUI.Toggle(SetMargin(rect, m), motionPath.editable, GUIContents.editPath, GUIStyles.toggleEdit);
                    }

                    rect.x += s;
                }
                else
                {
                    rect.x += s;

                    if (motionPath.HasCurveData)
                    {
                        motionPath.loop = GUI.Toggle(SetMargin(rect, m), motionPath.loop, GUIContents.loopPath, GUIStyles.toggleLoop);
                    }

                    rect.x += s;
                }
                GUI.enabled = true;

                motionPath.active = GUI.Toggle(SetMargin(rect, m), motionPath.active, GUIContents.activePath, GUIStyles.toggleVisibility);
            }

            private Rect SetMargin(Rect rect, float margin)
            {
                return new Rect(rect.x + margin, rect.y + margin, rect.width - margin, rect.height - margin);
            }

            private void DrawHeaderCallback(Rect rect)
            {

            }

            private bool OnCanAddCallback(ReorderableList list)
            {
                if (Selection.count == 0) return false;
                foreach (var transform in Selection.transforms)
                {
                    if (!m_List.Exists(e => e.transform == transform))
                    {
                        return true;
                    }
                }
                return false;
            }

            private void OnAddCallback(ReorderableList list)
            {
                foreach (var transform in Selection.transforms)
                {
                    if (!m_List.Exists(e => e.transform == transform))
                    {
                        m_List.Add(new MotionPath(transform));
                    }
                }
            }
        }
    }
}