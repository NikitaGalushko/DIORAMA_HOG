using System.Collections.Generic;
using UnityEngine;

namespace ScriptBoy.MotionPathAnimEditor
{

    static class HandleSelection
    {
        private static List<Handle> list = new List<Handle>();

        public static List<Handle> handles => list;

        public static Handle activeHandle => list[0];

        public static int count => list.Count;

        public static Vector3 center
        {
            get
            {
                if (count == 0) return Vector3.zero;

                Vector3 sum = Vector3.zero;
                foreach (var handle in list)
                {
                    sum += handle.position;
                }
                return sum / count;
            }
        }


        public static Quaternion rotation
        {
            get
            {
                if (count == 0) return Quaternion.identity;

                Quaternion q = activeHandle.matrix.rotation;
                foreach (var handle in handles)
                {
                    Quaternion q2 = handle.matrix.rotation;
                    if (q != q2) return Quaternion.identity;
                }
                return q;
            }
        }

        public static void Clear()
        {
            foreach (var item in list) item.selected = false;
            list.Clear();
        }

        public static bool Contains(Handle handle)
        {
            return list.Contains(handle);
        }

        public static void Remove(Handle handle)
        {
            list.Remove(handle);
            handle.selected = false;
        }

        public static void Add(Handle handle)
        {
            list.Add(handle);
            handle.selected = true;
        }

        public static void Select(Handle handle)
        {
            if (!Event.current.shift) Clear();

            if (Contains(handle))
            {
                Remove(handle);
            }
            else
            {
                Add(handle);
            }
        }

        public static void Select(List<Handle> handles)
        {
            if (!Event.current.shift) Clear();

            foreach (var handle in handles)
            {
                if (Contains(handle))
                {
                    Remove(handle);
                }
                else
                {
                    Add(handle);
                }
            }
        }
    }
}