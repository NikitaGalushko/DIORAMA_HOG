using System.Collections;
using System.Collections.Generic;
using System.Linq;
using UnityEditor;
using UnityEditor.PackageManager;
using UnityEngine;

namespace ScriptBoy.MotionPathAnimEditor
{
    static class DependencySolver
    {
        private static string[] dependencies = { "com.unity.settings-manager" };

        [InitializeOnLoadMethod]
        public static void Check()
        {
            var listRequest = Client.List(true);
            while (!listRequest.IsCompleted)
            {
                if ((listRequest.Status == StatusCode.Failure || listRequest.Error != null) && listRequest.Error != null)
                {
                    Debug.LogError(listRequest.Error.message);
                    return;
                }
            }

            for (int i = 0; i < dependencies.Length; i++)
            {
                string packageName = dependencies[i];

                if (listRequest.Result.FirstOrDefault((x) => x.name == packageName) == null)
                {
                    Debug.Log($"MotionPathAnimEditorDependencySolver: The package '{packageName}' is missing!");

                    var addRequest = Client.Add(packageName);
                    while (!addRequest.IsCompleted) { }

                    if (addRequest.Status == StatusCode.Success)
                    {
                        Debug.Log($"MotionPathAnimEditorDependencySolver: The package '{packageName}' is added!");
                    }
                    else
                    {
                        Debug.Log(listRequest.Error);
                    }
                }
            }
        }
    }
}