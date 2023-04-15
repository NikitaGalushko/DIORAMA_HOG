using UnityEditor;
using UnityEngine;

#if SETTINGS_MANAGER
using UnityEditor.SettingsManagement;
namespace ScriptBoy.MotionPathAnimEditor
{
    static partial class Settings
    {
        class UserSetting<T> : UnityEditor.SettingsManagement.UserSetting<T>
        {
            public UserSetting(string key, T value) : base(instance, key, value, SettingsScope.User) { }

            public T Value
            {
                get => value;
                set
                {
                    if (!Equals(value, this.value))
                        SetValue(value);
                }
            }
        }
    }
}
#else
namespace ScriptBoy.MotionPathAnimEditor
{
    static partial class Settings
    {
        class UserSetting<T>
        {
            public UserSetting(string key, T value)
            {
                m_Value = m_Default = value;
            }

            private T m_Value;
            private T m_Default;

            public T Value
            {
                get => m_Value;
                set => m_Value = value;
            }

            public void Reset()
            {
                m_Value = m_Default;
            }
        }
    }
}
#endif