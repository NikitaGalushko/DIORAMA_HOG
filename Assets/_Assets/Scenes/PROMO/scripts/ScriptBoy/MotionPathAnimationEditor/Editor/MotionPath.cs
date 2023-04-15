using System;
using System.Collections.Generic;
using UnityEditor;
using UnityEngine;
using TangentMode = UnityEditor.AnimationUtility.TangentMode;

namespace ScriptBoy.MotionPathAnimEditor
{
    class MotionPath
    {
        public MotionPath(Transform target)
        {
            m_Transform = target;
            m_ControlHandles = new List<ControlHandle>();
        }

        #region Variables
        private bool m_Active = true;
        private bool m_Editable;
        private bool m_Loop;

        private Transform m_Transform;

        private PositionCurveBinding m_PositionCurveBinding;

        private AnimationCurve m_XCurve;
        private AnimationCurve m_YCurve;
        private AnimationCurve m_ZCurve;

        private bool m_HasXAxis;
        private bool m_HasYAxis;
        private bool m_HasZAxis;

        private List<ControlHandle> m_ControlHandles;

        private List<Vector3> m_Path;
        private Dictionary<int, int> m_KeyIndexByFrame;
        private float[] m_KeyTimes;

        private float m_MinVelocity;
        private float m_MaxVelocity;
        #endregion

        #region Properties
        public Transform transform => m_Transform;

        public bool HasCurveData => m_HasXAxis || m_HasYAxis || m_HasZAxis;

        public bool IsEditable => m_Active && editable && HasCurveData;

        public string name
        {
            get
            {
                return m_Transform.name;
            }
        }

        public string fullName
        {
            get
            {
                string path;

                if (HasCurveData)
                {
                    path = m_PositionCurveBinding.path;
                }
                else
                {
                    path = AnimationUtility.CalculateTransformPath(m_Transform, AnimEditor.rootTransform);
                }

                path = AnimEditor.rootGameObject.name + ((path == "") ? "" : "/") + path;
                return path;
            }
        }

        public bool active
        {
            get => m_Active;
            set => m_Active = value;
        }

        public bool editable
        {
            get => m_Editable || !Settings.pathEditButton;
            set => m_Editable = value;
        }


        public bool loop
        {
            get => m_Loop;
            set
            {
                if (m_Loop != value)
                {
                    m_Loop = value;

                    if (m_Loop && m_ControlHandles.Count > 1)
                    {
                        m_ControlHandles[0].SetDirty();
                        ApplyChages();
                    }
                }
            }
        }

        private Matrix4x4 localToWorldMatrix
        {
            get
            {
                Matrix4x4 m;
                var parent = m_Transform.parent;
                m = parent == null ? Matrix4x4.identity : parent.localToWorldMatrix;

                if (Settings.applyRootOffset && transform == AnimEditor.rootTransform)
                {
                    m *= RootOffset.matrix;
                }

                return m;
            }
        }
        #endregion

        #region Methods
        public void SetAnimationCurves(PositionCurveBinding positionCurveBinding)
        {
            m_PositionCurveBinding = positionCurveBinding;

            m_HasXAxis = m_PositionCurveBinding.hasX;
            m_HasYAxis = m_PositionCurveBinding.hasY;
            m_HasZAxis = m_PositionCurveBinding.hasZ;

            if (m_HasXAxis) m_XCurve = AnimationUtility.GetEditorCurve(AnimEditor.animationClip, m_PositionCurveBinding.x);
            if (m_HasYAxis) m_YCurve = AnimationUtility.GetEditorCurve(AnimEditor.animationClip, m_PositionCurveBinding.y);
            if (m_HasZAxis) m_ZCurve = AnimationUtility.GetEditorCurve(AnimEditor.animationClip, m_PositionCurveBinding.z);
        }

        public void ClearAnimationCurves()
        {
            m_XCurve = m_YCurve = m_ZCurve = null;
        }

        public void UpdateHandlesMatrix()
        {
            Matrix4x4 matrix = localToWorldMatrix;
            int handleCount = m_ControlHandles.Count;
            for (int i = 0; i < handleCount; i++)
            {
                m_ControlHandles[i].SetMatrix(matrix);
            }
        }

        private void SetHandleCount(int count)
        {
            while (m_ControlHandles.Count < count)
            {
                m_ControlHandles.Add(new ControlHandle());
            }

            while (m_ControlHandles.Count > count)
            {
                m_ControlHandles.RemoveAt(m_ControlHandles.Count - 1);
            }
        }

        public void UpdateHandles()
        {
            if (!HasCurveData) return;
            Keyframe[] xKeys = m_HasXAxis ? m_XCurve.keys : null;
            Keyframe[] yKeys = m_HasYAxis ? m_YCurve.keys : null;
            Keyframe[] zKeys = m_HasZAxis ? m_ZCurve.keys : null;

            int keyCount = m_XCurve.length;
            SetHandleCount(keyCount);

            for (int i = 0; i < keyCount; i++)
            {
                float prevTime = xKeys[LoopUtility.Mod(i - 1, keyCount)].time;
                float currentTime = xKeys[i].time;
                float nextTime = xKeys[LoopUtility.Mod(i + 1, keyCount)].time;
                float leftTangentScale = (prevTime - currentTime) / 3;
                float rightTangentScale = (currentTime - nextTime) / 3;

                leftTangentScale = -Mathf.Abs(leftTangentScale);
                rightTangentScale = -Mathf.Abs(rightTangentScale);

                if (i == 0)
                {
                    float a = xKeys[LoopUtility.Mod(i - 1, keyCount)].time;
                    float b = xKeys[LoopUtility.Mod(i - 2, keyCount)].time;

                    leftTangentScale = (b - a) / 3;
                }

                Vector3 handlePosition;
                handlePosition.x = m_HasXAxis ? xKeys[i].value : 0;
                handlePosition.y = m_HasYAxis ? yKeys[i].value : 0;
                handlePosition.z = m_HasZAxis ? zKeys[i].value : 0;

                Vector3 inTangents;
                inTangents.x = m_HasXAxis ? xKeys[i].inTangent : 0; 
                inTangents.y = m_HasYAxis ? yKeys[i].inTangent : 0;
                inTangents.z = m_HasZAxis ? zKeys[i].inTangent : 0;

                Vector3 outTangents;
                outTangents.x = m_HasXAxis ? xKeys[i].outTangent : 0;
                outTangents.y = m_HasYAxis ? yKeys[i].outTangent : 0;
                outTangents.z = m_HasZAxis ? zKeys[i].outTangent : 0;


                float inTangentsM = inTangents.magnitude;
                float outTangentsM = outTangents.magnitude;
                float tangentsRaito = inTangentsM / outTangentsM;
                bool hasParallelTangents = inTangents / inTangentsM == outTangents / outTangentsM;


                inTangents = handlePosition + inTangents * leftTangentScale;
                outTangents = handlePosition - outTangents * rightTangentScale;

                ControlHandle handle = m_ControlHandles[i];
                TangentHandle tangentLeft = handle.leftTangent;
                TangentHandle tangentRight = handle.rightTangent;

                handle.localPosition = handlePosition;
                tangentLeft.localPosition = inTangents;
                tangentRight.localPosition = outTangents;
                tangentLeft.scale = leftTangentScale;
                tangentRight.scale = rightTangentScale;
                tangentLeft.mode = AnimationUtility.GetKeyLeftTangentMode(m_XCurve, i);
                tangentRight.mode = AnimationUtility.GetKeyRightTangentMode(m_XCurve, i);
                tangentLeft.hide = i == 0;
                tangentRight.hide = i == keyCount - 1;
                handle.hide = false;
                handle.hasChanged = false;
                tangentLeft.hasChanged = false;
                tangentRight.hasChanged = false;
                handle.tangentsRaito = tangentsRaito;
                handle.hasParallelTangents = hasParallelTangents;
            }

            if (m_Loop)
            {
                var start = m_ControlHandles[0];
                var end = m_ControlHandles[keyCount - 1];
                start.leftTangent.hide = false;
                end.hide = true;
                end.position = start.position;


                if (end.leftTangent.mode != start.leftTangent.mode)
                {
                    end.leftTangent.mode = start.leftTangent.mode;
                    SetTangantsMode(keyCount - 1, start.leftTangent.mode, true, true);
                    end.leftTangent.position = start.leftTangent.position;
                    start.leftTangent.hasChanged = true;
                    GUI.changed = true;
                }

                if (end.leftTangent.mode == TangentMode.Free)
                    end.leftTangent.position = start.leftTangent.position;

                if (end.rightTangent.mode == TangentMode.Free)
                    end.rightTangent.position = start.rightTangent.position;
            }
        }

        public void ApplyChages()
        {
            if (!HasCurveData) return;

            Keyframe[] xKeys = m_HasXAxis ? m_XCurve.keys : null;
            Keyframe[] yKeys = m_HasYAxis ? m_YCurve.keys : null;
            Keyframe[] zKeys = m_HasZAxis ? m_ZCurve.keys : null;

            Event EVENT = Event.current;
            bool alt = EVENT.alt;
            bool restTangents = alt && HandleSelection.count < 2;


            int handleCount = m_ControlHandles.Count;

            if (m_Loop)
            {
                var start = m_ControlHandles[0];
                var end = m_ControlHandles[handleCount - 1];

                end.hasChanged = start.hasChanged;
                end.position = start.position;

                end.leftTangent.hasChanged = start.leftTangent.hasChanged;
                end.leftTangent.position = start.leftTangent.position;
                end.leftTangent.scale = start.leftTangent.scale;

                end.rightTangent.hasChanged = start.rightTangent.hasChanged;
                end.rightTangent.position = start.rightTangent.position;
                end.rightTangent.scale = start.rightTangent.scale;
            }

            for (int i = 0; i < handleCount; i++)
            {
                ControlHandle handle = m_ControlHandles[i];
                if (handle.hasChanged)
                {
                    Vector3 local = handle.localPosition;
                    if (restTangents)
                    {
                        Vector3 old = new Vector3();
                        if (m_HasXAxis) old.x = xKeys[i].value;
                        if (m_HasYAxis) old.y = yKeys[i].value;
                        if (m_HasZAxis) old.z = zKeys[i].value;

                        if (handle.leftTangent.mode == TangentMode.Free)
                        {
                            Vector3 inTangent = (local - old) / handle.leftTangent.scale;
                            if (m_HasXAxis) xKeys[i].inTangent = inTangent.x;
                            if (m_HasYAxis) yKeys[i].inTangent = inTangent.y;
                            if (m_HasZAxis) zKeys[i].inTangent = inTangent.z;
                        }

                        if (handle.rightTangent.mode == TangentMode.Free)
                        {
                            Vector3 outTangent = (local - old) / handle.leftTangent.scale;
                            if (m_HasXAxis) xKeys[i].outTangent = outTangent.x;
                            if (m_HasYAxis) yKeys[i].outTangent = outTangent.y;
                            if (m_HasZAxis) zKeys[i].outTangent = outTangent.z;
                        }

                        continue;
                    }

                    if (m_HasXAxis) xKeys[i].value = local.x;
                    if (m_HasYAxis) yKeys[i].value = local.y;
                    if (m_HasZAxis) zKeys[i].value = local.z;
                }

                if (handle.leftTangent.hasChanged)
                {
                    Vector3 local = handle.leftTangent.localPosition - handle.localPosition;
                    local /= handle.leftTangent.scale;


                    if (m_HasXAxis) xKeys[i].inTangent = local.x;
                    if (m_HasYAxis) yKeys[i].inTangent = local.y;
                    if (m_HasZAxis) zKeys[i].inTangent = local.z;


                    if (!alt && handle.hasParallelTangents)
                    {
                        if (m_HasXAxis) xKeys[i].outTangent = local.x / handle.tangentsRaito;
                        if (m_HasYAxis) yKeys[i].outTangent = local.y / handle.tangentsRaito;
                        if (m_HasZAxis) zKeys[i].outTangent = local.z / handle.tangentsRaito;
                    }
                }

                if (handle.rightTangent.hasChanged)
                {
                    Vector3 local = handle.localPosition - handle.rightTangent.localPosition;
                    local /= handle.rightTangent.scale;


                    if (m_HasXAxis) xKeys[i].outTangent = local.x;
                    if (m_HasYAxis) yKeys[i].outTangent = local.y;
                    if (m_HasZAxis) zKeys[i].outTangent = local.z;


                    if (!alt && handle.hasParallelTangents)
                    {
                        float raito = 1 / handle.tangentsRaito;
                        if (m_HasXAxis) xKeys[i].inTangent = local.x / raito;
                        if (m_HasYAxis) yKeys[i].inTangent = local.y / raito;
                        if (m_HasZAxis) zKeys[i].inTangent = local.z / raito;
                    }
                }
            }

            UpdateClip(xKeys, yKeys, zKeys);
        }

        private void UpdateClip(Keyframe[] xKeys, Keyframe[] yKeys, Keyframe[] zKeys)
        {
            if (m_HasXAxis) m_XCurve.keys = xKeys;
            if (m_HasYAxis) m_YCurve.keys = yKeys;
            if (m_HasZAxis) m_ZCurve.keys = zKeys;

            RefreshTangents();
            UpdateClip();
        }

        private void UpdateClip()
        {
            Undo.RecordObject(AnimEditor.animationClip, "Edit Curves");

            if (m_HasXAxis) AnimationUtility.SetEditorCurve(AnimEditor.animationClip, m_PositionCurveBinding.x, m_XCurve);
            if (m_HasYAxis) AnimationUtility.SetEditorCurve(AnimEditor.animationClip, m_PositionCurveBinding.y, m_YCurve);
            if (m_HasZAxis) AnimationUtility.SetEditorCurve(AnimEditor.animationClip, m_PositionCurveBinding.z, m_ZCurve);
        }

        private void RefreshTangents()
        {
            int keyCount = m_XCurve.length;
            for (int i = 0; i < keyCount; i++)
            {
                TangentMode leftTangentMode = AnimationUtility.GetKeyLeftTangentMode(m_XCurve, i);
                TangentMode rightTangentMode = AnimationUtility.GetKeyRightTangentMode(m_XCurve, i);

                bool broken = leftTangentMode != rightTangentMode || leftTangentMode != TangentMode.Auto && leftTangentMode != TangentMode.ClampedAuto;

                if (m_HasXAxis)
                {
                    AnimationUtility.SetKeyLeftTangentMode(m_XCurve, i, leftTangentMode);
                    AnimationUtility.SetKeyRightTangentMode(m_XCurve, i, rightTangentMode);
                    AnimationUtility.SetKeyBroken(m_XCurve, i, broken);
                }

                if (m_HasYAxis)
                {
                    AnimationUtility.SetKeyLeftTangentMode(m_YCurve, i, leftTangentMode);
                    AnimationUtility.SetKeyRightTangentMode(m_YCurve, i, rightTangentMode);
                    AnimationUtility.SetKeyBroken(m_YCurve, i, broken);
                }

                if (m_HasZAxis)
                {
                    AnimationUtility.SetKeyLeftTangentMode(m_ZCurve, i, leftTangentMode);
                    AnimationUtility.SetKeyRightTangentMode(m_ZCurve, i, rightTangentMode);
                    AnimationUtility.SetKeyBroken(m_ZCurve, i, broken);
                }
            }
        }


        private void CalcVelocityRange()
        {
            if (Event.current.type != EventType.Repaint) return;

            int count = m_ControlHandles.Count;
            if (count < 2) return;

            int segmentCount = (int)Settings.pathAccuracy + 1;
            float minDis = float.PositiveInfinity;
            m_MinVelocity = float.PositiveInfinity;
            m_MaxVelocity = float.NegativeInfinity;

            for (int i = 1; i < count; i++)
            {
                ControlHandle startHandle = m_ControlHandles[i - 1];
                ControlHandle endHandle = m_ControlHandles[i];

                Vector3 start = startHandle.position;
                Vector3 end = endHandle.position;

                Vector3 startTangent = startHandle.rightTangent.position;
                Vector3 endTangent = endHandle.leftTangent.position;


                if (float.IsInfinity(startTangent.x) || float.IsInfinity(endTangent.x))
                {
                    continue;
                }
                else
                {
                    float deltaTime = startHandle.rightTangent.scale;
                    Vector3 a = start;
                    for (int j = 1; j <= segmentCount; j++)
                    {
                        float t = (float)j / segmentCount;
                        Vector3 b = BezierCurveRenderer.EvaluateBezierCurve(start, end, startTangent, endTangent, t);
                        float velocity = (b - a).magnitude / deltaTime;
                        m_MinVelocity = Mathf.Min(m_MinVelocity, velocity);
                        m_MaxVelocity = Mathf.Max(m_MaxVelocity, velocity);
                        float d = HandleUtility.DistanceToLine(a, b);
                        if (minDis > d) minDis = d;
                        a = b;
                    }
                }
            }
        }

        public void FixMissingKeyframes()
        {
            int axisCount = 0;

            if (m_HasXAxis) axisCount++;
            if (m_HasYAxis) axisCount++;
            if (m_HasZAxis) axisCount++;

            //000 001 010 100
            if (axisCount < 2) return;

            int xCount = 0;
            int yCount = 0;
            int zCount = 0;

            if (m_HasXAxis) xCount = m_XCurve.length;
            if (m_HasYAxis) yCount = m_YCurve.length;
            if (m_HasZAxis) zCount = m_ZCurve.length;
            
            //011
            if (!m_HasXAxis && m_HasYAxis && m_HasZAxis && yCount == zCount) return;
            //101
            if (m_HasXAxis && !m_HasYAxis && m_HasZAxis && xCount == zCount) return;
            //110
            if (m_HasXAxis && m_HasYAxis && !m_HasZAxis && xCount == yCount) return;
            //111
            if (m_HasXAxis && m_HasYAxis && m_HasZAxis && xCount == yCount && xCount == zCount && yCount == zCount) return;

            Keyframe[] xKeys = m_HasXAxis ? m_XCurve.keys : null;
            Keyframe[] yKeys = m_HasYAxis ? m_YCurve.keys : null;
            Keyframe[] zKeys = m_HasZAxis ? m_ZCurve.keys : null;

            HashSet<float> times = new HashSet<float>();

            foreach (var key in xKeys) times.Add(key.time);
            foreach (var key in yKeys) times.Add(key.time);
            foreach (var key in zKeys) times.Add(key.time);

            foreach (var time in times)
            {
                bool exist;

                if (m_HasXAxis)
                {
                    exist = false;
                    foreach (var key in xKeys) exist |= key.time == time;
                    if (!exist) m_XCurve.AddKey(time, m_XCurve.Evaluate(time));
                }

                if (m_HasYAxis)
                {
                    exist = false;
                    foreach (var key in yKeys) exist |= key.time == time;
                    if (!exist) m_YCurve.AddKey(time, m_YCurve.Evaluate(time));
                }

                if (m_HasZAxis)
                {
                    exist = false;
                    foreach (var key in zKeys) exist |= key.time == time;
                    if (!exist) m_ZCurve.AddKey(time, m_ZCurve.Evaluate(time));
                }
            }

            UpdateClip();
        }

        public Vector3 GetPositionAtTime(float time)
        {
            float x = 0;
            float y = 0;
            float z = 0;

            if (m_HasXAxis) x = m_XCurve.Evaluate(time);
            if (m_HasYAxis) y = m_YCurve.Evaluate(time);
            if (m_HasZAxis) z = m_ZCurve.Evaluate(time);

            return new Vector3(x,y,z);
        }

        #region Cache World Path
        public void StartCachingWorldPath(int frameCount, float time2Frame, HashSet<int> hotFrames)
        {
            if (m_Path == null)
            {
                m_Path = new List<Vector3>();
            }
            else
            {
                m_Path.Clear();
            }

            if (m_KeyIndexByFrame == null) m_KeyIndexByFrame = new Dictionary<int, int>();

            if (!HasCurveData) return;

            Keyframe[] keys = m_XCurve.keys;
            int keyCount = keys.Length;
            //bool timeChanged = false;

            if (m_KeyTimes == null || m_KeyTimes.Length != keyCount)
            {
                m_KeyIndexByFrame.Clear();
                m_KeyTimes = new float[keyCount];
                //timeChanged = true;
                for (int i = 0; i < keyCount; i++)
                {
                    float t = keys[i].time;
                    int f = (int)(t * time2Frame);

                    m_KeyTimes[i] = t;
                    m_KeyIndexByFrame.Add(f, i);
                    hotFrames.Add(f);
                }
            }
            else
            {
                for (int i = 0; i < keyCount; i++)
                {
                    float prev = m_KeyTimes[i];
                    float current = keys[i].time;
                    int f = (int)(current * time2Frame);
                    if (prev != current)
                    {
                        m_KeyIndexByFrame.Remove((int)(prev * time2Frame));
                        m_KeyIndexByFrame.Add((int)(current * time2Frame), i);
                        m_KeyTimes[i] = current;
                        //timeChanged = true;
                    }
                    hotFrames.Add(f);
                }
            }
        }

        public void CacheWorldPosition(int frameIndex, int frameCount)
        {
            Vector3 po = m_Transform.position;
            int pCount = m_Path.Count;
            if (pCount == 0 || frameIndex == frameCount)
            {
                m_Path.Add(po);
            }
            else
            {
                Vector3 prev = m_Path[m_Path.Count - 1];
                if ((prev - po).sqrMagnitude > 0.0001f)
                {
                    m_Path.Add(po);
                }
            }

            if (m_KeyIndexByFrame.TryGetValue(frameIndex, out int keyIndex) && m_ControlHandles.Count > keyIndex)
            {
                m_ControlHandles[keyIndex].SetMatrix(localToWorldMatrix);
            }
        }
        #endregion

        #region Draw Path/Curve
        public void DrawPath()
        {
            Handles.color = Settings.pathColor;
            Handles.DrawAAPolyLine(4, m_Path.ToArray());
        }

        public void DrawCurves()
        {
            if (!HasCurveData) return;
            if (Event.current.type != EventType.Repaint) return;

            int count = m_ControlHandles.Count;
            if (count < 2) return;

            CalcVelocityRange();

            ColorMode curveDrawMode = Settings.pathColorMode;
            float curveAccuracy = Settings.pathAccuracy;

            for (int i = 1; i < count; i++)
            {
                ControlHandle startHandle = m_ControlHandles[i - 1];
                ControlHandle endHandle = m_ControlHandles[i];

                Vector3 start = startHandle.position;
                Vector3 end = endHandle.position;

                Vector3 startTangent = startHandle.rightTangent.position;
                Vector3 endTangent = endHandle.leftTangent.position;

                if (float.IsInfinity(startTangent.x) || float.IsInfinity(endTangent.x))
                {
                    Handles.color = Settings.pathColor;
                    Handles.DrawDottedLine(start, end, 3f);
                }
                else
                {
                    if (curveDrawMode == ColorMode.Gradient)
                    {
                        float deltaTime = startHandle.rightTangent.scale;
                        BezierCurveRenderer.Begin();
                        BezierCurveRenderer.Draw(start, end, startTangent, endTangent, deltaTime, m_MinVelocity, m_MaxVelocity, (int)curveAccuracy + 1);
                        BezierCurveRenderer.End();
                    }
                    else
                    {
                        Handles.DrawBezier(start, end, startTangent, endTangent, Settings.pathColor, null, 5);
                    }
                }
            }
        }
        #endregion

        #region Editor Handles
        public void DrawHandlesCap()
        {
            foreach (var handle in m_ControlHandles)
            {
                handle.DrawHandlesCap();
            }
        }

        public void DrawSelectionButtons()
        {
            Event EVENT = Event.current;
            bool mouseRightClick = EVENT.isMouse && EVENT.type == EventType.MouseDown && EVENT.button == 1;


            foreach (var handle in m_ControlHandles)
            {
                if (handle.DrawSelectableButtons())
                {
                    if (mouseRightClick) OpenHandleMenu(m_ControlHandles.IndexOf(handle));
                }
            }
        }

        public void EditCurves2D()
        {
            if (Event.current.shift)
            {
                DrawSelectionButtons();
            }
            else
            {
                MouseRecords.Record();

                int handleCount = m_ControlHandles.Count;
                for (int i = 0; i < handleCount; i++)
                {
                    if (m_ControlHandles[i].DoFreeMoveHandles())
                    {
                        if (MouseRecords.RightClick) OpenHandleMenu(i);
                    }
                }
            }
        }

        public void EditCurves3D()
        {
            Event EVENT = Event.current;
            bool control = EVENT.control;
            bool mouseClickRight = EVENT.isMouse && EVENT.type == EventType.MouseDown && EVENT.button == 1;
            bool mouseClickLeft = EVENT.isMouse && EVENT.type == EventType.MouseDown && EVENT.button == 0;
            int handleCount = m_ControlHandles.Count;

            DrawSelectionButtons();

            if (HandleSelection.count == 1)
            {
                HandleSelection.activeHandle.DoPositionHandle();
            }
        }

        public void CheckBoxSelection()
        {
            foreach (var handle in m_ControlHandles)
            {
                handle.CheckBoxSelection();
            }
        }
        #endregion

        #region Handle Menu
        private void OpenHandleMenu(int handleIndex)
        {
            GenericMenu menu = new GenericMenu();

            TangentMode leftTangentMode = AnimationUtility.GetKeyLeftTangentMode(m_XCurve, handleIndex);
            TangentMode rightTangentMode = AnimationUtility.GetKeyRightTangentMode(m_XCurve, handleIndex);

            bool isLClampedAuto = leftTangentMode == TangentMode.ClampedAuto;
            bool isLAuto = leftTangentMode == TangentMode.Auto;
            bool isLFree = leftTangentMode == TangentMode.Free;
            bool isLLinear = leftTangentMode == TangentMode.Linear;
            bool isLConstant = leftTangentMode == TangentMode.Constant;

            bool isRClampedAuto = rightTangentMode == TangentMode.ClampedAuto;
            bool isRAuto = rightTangentMode == TangentMode.Auto;
            bool isRFree = rightTangentMode == TangentMode.Free;
            bool isRLinear = rightTangentMode == TangentMode.Linear;
            bool isRConstant = rightTangentMode == TangentMode.Constant;

            bool areLRClampedAuto = isLClampedAuto && isRClampedAuto;
            bool areLRAuto = isLAuto && isRAuto;
            bool areLRFree = isLFree && isRFree;
            bool areLRLinear = isLLinear && isRLinear;
            bool areLRConstant = isLConstant && isRConstant;

            bool broken = leftTangentMode != rightTangentMode || leftTangentMode != TangentMode.Auto && leftTangentMode != TangentMode.ClampedAuto;

            menu.AddItem(new GUIContent("Delete"), false, () => DeleteHandle(handleIndex));

            menu.AddSeparator("");

            menu.AddItem(new GUIContent("Clamped Auto"), areLRClampedAuto, () => SetTangantsMode(handleIndex, TangentMode.ClampedAuto));
            menu.AddItem(new GUIContent("Auto"), areLRAuto, () => SetTangantsMode(handleIndex, TangentMode.Auto));
            menu.AddItem(new GUIContent("Broken"), broken, () => SetTangantsMode(handleIndex, TangentMode.Free));

            menu.AddSeparator("");

            menu.AddItem(new GUIContent("Left Tangent/Free"), isLFree, () => SetLeftTangantMode(handleIndex, TangentMode.Free));
            menu.AddItem(new GUIContent("Left Tangent/Liner"), isLLinear, () => SetLeftTangantMode(handleIndex, TangentMode.Linear));
            menu.AddItem(new GUIContent("Left Tangent/Constant"), isLConstant, () => SetLeftTangantMode(handleIndex, TangentMode.Constant));

            menu.AddItem(new GUIContent("Right Tangent/Free"), isRFree, () => SetRightTangantMode(handleIndex, TangentMode.Free));
            menu.AddItem(new GUIContent("Right Tangent/Liner"), isRLinear, () => SetRightTangantMode(handleIndex, TangentMode.Linear));
            menu.AddItem(new GUIContent("Right Tangent/Constant"), isRConstant, () => SetRightTangantMode(handleIndex, TangentMode.Constant));

            menu.AddItem(new GUIContent("Both Tangents/Free"), areLRFree, () => SetTangantsMode(handleIndex, TangentMode.Free));
            menu.AddItem(new GUIContent("Both Tangents/Liner"), areLRLinear, () => SetTangantsMode(handleIndex, TangentMode.Linear));
            menu.AddItem(new GUIContent("Both Tangents/Constant"), areLRConstant, () => SetTangantsMode(handleIndex, TangentMode.Constant));

            menu.ShowAsContext();
        }

        private void DeleteHandle(int handleIndex)
        {
            if (m_HasXAxis) m_XCurve.RemoveKey(handleIndex);
            if (m_HasYAxis) m_YCurve.RemoveKey(handleIndex);
            if (m_HasZAxis) m_ZCurve.RemoveKey(handleIndex);

            UpdateClip();
        }

        private void SetTangantsMode(int handleIndex, TangentMode tangentMode, bool inTangent = true, bool outTangent = true)
        {
            if (inTangent)
            {
                if (m_HasXAxis) AnimationUtility.SetKeyLeftTangentMode(m_XCurve, handleIndex, tangentMode);
                if (m_HasYAxis) AnimationUtility.SetKeyLeftTangentMode(m_YCurve, handleIndex, tangentMode);
                if (m_HasZAxis) AnimationUtility.SetKeyLeftTangentMode(m_ZCurve, handleIndex, tangentMode);
            }

            if (outTangent)
            {
                if (m_HasXAxis) AnimationUtility.SetKeyRightTangentMode(m_XCurve, handleIndex, tangentMode);
                if (m_HasYAxis) AnimationUtility.SetKeyRightTangentMode(m_YCurve, handleIndex, tangentMode);
                if (m_HasZAxis) AnimationUtility.SetKeyRightTangentMode(m_ZCurve, handleIndex, tangentMode);
            }

            UpdateClip();
        }

        private void SetLeftTangantMode(int handleIndex, TangentMode tangentMode)
        {
            SetTangantsMode(handleIndex, tangentMode, true, false);

            if (tangentMode != TangentMode.Auto && tangentMode != TangentMode.ClampedAuto)
            {
                TangentMode rightTangentMode = AnimationUtility.GetKeyRightTangentMode(m_XCurve, handleIndex);
                if (rightTangentMode == TangentMode.Auto || rightTangentMode == TangentMode.ClampedAuto)
                {
                    SetTangantsMode(handleIndex, TangentMode.Free, false, true);
                }
            }
        }

        private void SetRightTangantMode(int handleIndex, TangentMode tangentMode)
        {
            SetTangantsMode(handleIndex, tangentMode, false, true);

            if (tangentMode != TangentMode.Auto && tangentMode != TangentMode.ClampedAuto)
            {
                TangentMode leftTangentMode = AnimationUtility.GetKeyLeftTangentMode(m_XCurve, handleIndex);
                if (leftTangentMode == TangentMode.Auto || leftTangentMode == TangentMode.ClampedAuto)
                {
                    SetTangantsMode(handleIndex, TangentMode.Free, true, false);
                }
            }
        }
        #endregion
        #endregion
    }
}