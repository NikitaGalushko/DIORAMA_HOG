using System;
using UnityEditor;
using UnityEngine;
using TangentMode = UnityEditor.AnimationUtility.TangentMode;

namespace ScriptBoy.MotionPathAnimEditor
{
    abstract class Handle
    {
        public Matrix4x4 matrix;
        public Vector3 localPosition;
        public Vector3 position
        {
            get => matrix.MultiplyPoint(localPosition);
            set => localPosition = matrix.inverse.MultiplyPoint(value);
        }

        public Quaternion rotation = Quaternion.identity;

        public bool hasChanged;
        public bool selected;
        public bool hide;

        protected Color color => GetColor(selected);
        protected Color GetColor(bool selected) => selected ? Settings.handleColorSelected : Settings.handleColorNormal;
 
        protected abstract HandleCapShape capShape { get; }

        public Handles.CapFunction capFunction
        {
            get
            {
                switch (capShape)
                {
                    default:
                    case HandleCapShape.Cube: return Handles.CubeHandleCap;
                    case HandleCapShape.Sphere: return Handles.SphereHandleCap;
                    case HandleCapShape.Cone: return Handles.ConeHandleCap;
                }
            }
        }

        protected float GetSize()
        {
            return HandleUtility.GetHandleSize(position) * Settings.handleSize;
        }

        public void DoPositionHandle()
        {
            Quaternion q = Tools.pivotRotation == PivotRotation.Global ? Quaternion.identity : matrix.rotation;
            EditorGUI.BeginChangeCheck();
            var newPosition = Handles.DoPositionHandle(position, q);
            if (EditorGUI.EndChangeCheck())
            {
                newPosition = EditorGridUtility.SnapToGrid(newPosition);
                hasChanged = position != newPosition;
                position = newPosition;
            }
        }

        Quaternion r;
        public bool FreeMoveHandle()
        {
            Handles.color = color;
            float size = GetSize();
            EditorGUI.BeginChangeCheck();
            var newPosition = Handles.FreeMoveHandle(position, rotation, size, Vector3.zero, HandleCap);
            if (EditorGUI.EndChangeCheck())
            {
                if (Settings.localSnappingIn2D)
                {
                    newPosition = EditorGridUtility.SnapToLocalGrid(newPosition, matrix);
                    hasChanged = position != newPosition;
                    position = newPosition;
                    GridRenderer.SetDefaultMatrix(matrix, Vector3.zero);
                }
                else
                {
                    newPosition = EditorGridUtility.SnapToGrid(newPosition);
                    hasChanged = position != newPosition;
                    position = newPosition;
                }
            }

            if (HandleUtility.DistanceToCircle(position, size) == 0)
            {
                if (MouseRecords.LeftClick)
                {
                    HandleSelection.Select(this);
                    AnimEditorWindow.RepaintWindow();
                }

                return true;
            }

            return false;
        }

        private void HandleCap(int controlID, Vector3 position, Quaternion _rotation, float size, EventType eventType)
        {
           capFunction.Invoke(controlID, position,rotation, size, eventType);
        }

        public void DrawButton()
        {
            Handles.color = color;
            if (Handles.Button(position, rotation, GetSize(), 1, capFunction))
            {
                HandleSelection.Select(this);
                AnimEditorWindow.RepaintWindow();
            }
        }

        public void DrawHandleCap()
        {
            Handles.color = color;
            capFunction.Invoke(0, position, rotation, GetSize(), EventType.Repaint);
        }
    }

    class ControlHandle : Handle
    {
        public TangentHandle leftTangent;
        public TangentHandle rightTangent;

        public bool hasParallelTangents;
        public float tangentsRaito;

        public bool hasLeftTangent => leftTangent.mode == TangentMode.Free && !leftTangent.hide;
        public bool hasRightTangent => rightTangent.mode == TangentMode.Free && !rightTangent.hide;


        protected override HandleCapShape capShape => Settings.handleCapControl;

        public ControlHandle()
        {
            leftTangent = new TangentHandle();
            rightTangent = new TangentHandle();
        }

        public bool DoFreeMoveHandles()
        {
            if (hide) return false;

            if (hasLeftTangent)
            {
                Handles.color = GetColor(selected && leftTangent.selected);
                Handles.DrawLine(position, leftTangent.position);
                leftTangent.rotation = LookRotation(position - leftTangent.position);
                leftTangent.FreeMoveHandle();
            }

            if (hasRightTangent)
            {
                Handles.color = GetColor(selected && rightTangent.selected);
                Handles.DrawLine(position, rightTangent.position);
                rightTangent.rotation = LookRotation(rightTangent.position - position);
                rightTangent.FreeMoveHandle();
            }

            return FreeMoveHandle();
        }

        public bool DrawSelectableButtons()
        {
            if (hide) return false;

            if (hasLeftTangent)
            {
                Handles.color = GetColor(selected && leftTangent.selected);
                Handles.DrawLine(position, leftTangent.position);

                leftTangent.rotation = LookRotation(position - leftTangent.position);
                leftTangent.DrawButton();
            }

            if (hasRightTangent)
            {
                Handles.color = GetColor(selected && rightTangent.selected);
                Handles.DrawLine(position, rightTangent.position);

                rightTangent.rotation = LookRotation(rightTangent.position - position);
                rightTangent.DrawButton();
            }

            DrawButton();

            return HandleUtility.DistanceToCircle(position, GetSize()) == 0;
        }

        public void DrawHandlesCap()
        {
            if (hide) return;

            if (hasLeftTangent)
            {
                Handles.color = GetColor( selected && leftTangent.selected);
                Handles.DrawLine(position, leftTangent.position);
                leftTangent.DrawHandleCap();
            }

            if (hasRightTangent)
            {
                Handles.color = GetColor(selected && rightTangent.selected);
                Handles.DrawLine(position, rightTangent.position);
                rightTangent.DrawHandleCap();
            }

            DrawHandleCap();
        }

        public void CheckBoxSelection()
        {
                  if (hide) return;

            if (hasLeftTangent) HandleSelectionRect.Check(leftTangent);
            if (hasRightTangent) HandleSelectionRect.Check(rightTangent);
            HandleSelectionRect.Check(this);
        }

        public void SetMatrix(Matrix4x4 matrix)
        {
            this.matrix = matrix;
            leftTangent.matrix = matrix;
            rightTangent.matrix = matrix;
        }

        public void SetDirty()
        {
            hasChanged = leftTangent.hasChanged = rightTangent.hasChanged = true;
        }

        private Quaternion LookRotation(Vector3 dir)
        {
            if (dir == Vector3.zero) return Quaternion.identity;
            return Quaternion.LookRotation(dir);
        }
    }


    class TangentHandle : Handle
    {
        public TangentMode mode;
        protected override HandleCapShape capShape => Settings.handleCapTangent;
        public float scale;
    }
}