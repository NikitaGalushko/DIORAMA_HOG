using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class AiController : MonoBehaviour
{
    public UnityEngine.AI.NavMeshAgent agent;

    [Range(0, 100)]public float speed;
    [Range(1, 500)]public float walkRadius;

    public void Start()
    {
    agent = GetComponent<UnityEngine.AI.NavMeshAgent>();
    if(agent !=null) 
        {
            agent.speed = speed;
            agent.SetDestination(RandomNavMeshLocation());
        }

    }

    public void Update()
    {
        if (agent != null && agent.remainingDistance <= agent.stoppingDistance)
        {
            agent.SetDestination(RandomNavMeshLocation());
        }
    }


    public Vector3 RandomNavMeshLocation()
    {
        Vector3 finalPosition = Vector3.zero;
        Vector3 randomPosition = Random.insideUnitSphere * walkRadius;
        randomPosition += transform.position;
        if(UnityEngine.AI.NavMesh.SamplePosition(randomPosition, out UnityEngine.AI.NavMeshHit hit,walkRadius, 1))
        {
            finalPosition = hit.position;    
        }
        return finalPosition;
    }
}
