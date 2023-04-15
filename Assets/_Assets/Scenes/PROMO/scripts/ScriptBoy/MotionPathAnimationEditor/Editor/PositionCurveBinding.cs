using System.Collections;
using UnityEditor;
using UnityEngine;

namespace ScriptBoy.MotionPathAnimEditor
{
    class PositionCurveBinding
    {
        public string path;
        public Transform transform;

        private EditorCurveBinding m_X;
        private EditorCurveBinding m_Y;
        private EditorCurveBinding m_Z;

        public EditorCurveBinding x
        {
            get
            {
                if (!hasX) throw new UnassignedFieldException("x");
                return m_X;
            }
            set
            {
                m_X = value;
                hasX = true;
            }
        }

        public EditorCurveBinding y
        {
            get
            {
                if (!hasY) throw new UnassignedFieldException("y");
                return m_Y;
            }
            set
            {
                m_Y = value;
                hasY = true;
            }
        }


        public EditorCurveBinding z
        {
            get
            {
                if (!hasZ) throw new UnassignedFieldException("z");
                return m_Z;
            }
            set
            {
                m_Z = value;
                hasZ = true;
            }
        }

        public bool hasX;
        public bool hasY;
        public bool hasZ;
    }

    internal class UnassignedFieldException : System.Exception
    {
        public UnassignedFieldException(string name) : base($"The variable ({name}) has been not assigned!")
        {

        }
    }
}