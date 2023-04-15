using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class ObjectHider : MonoBehaviour
{
    private void OnMouseDown()
    {
        gameObject.SetActive(false);
    }
}
