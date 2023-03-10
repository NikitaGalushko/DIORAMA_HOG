using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEditor;
using UnityEditor.Rendering.PostProcessing;

namespace CR
{
	[PostProcessEditor(typeof(CleanOutline))]
	public class CleanOutlineEditor : PostProcessEffectEditor<CleanOutline>
	{
		SerializedParameterOverride m_DebugMode;
		SerializedParameterOverride m_OutlineThickness;
		SerializedParameterOverride m_OutlineColor;
		
		SerializedParameterOverride m_EnableClosenessBoost;
		SerializedParameterOverride m_ClosenessBoostThickness;
		SerializedParameterOverride m_ClosenessBoostNear;
		SerializedParameterOverride m_ClosenessBoostFar;
		SerializedParameterOverride m_EnableDistanceFade;
		SerializedParameterOverride m_DistanceFadeNear;
		SerializedParameterOverride m_DistanceFadeFar;
		SerializedParameterOverride m_Depth9TilesThreshold;
		SerializedParameterOverride m_Depth9TilesBottomFix;
		SerializedParameterOverride m_DepthThreshold;

		SerializedParameterOverride m_DepthSampleType;
		SerializedParameterOverride m_DepthThickness;
		SerializedParameterOverride m_DepthMultiplier;
		SerializedParameterOverride m_DepthBias;

		SerializedParameterOverride m_EnableNormalOutline;
		SerializedParameterOverride m_NormalThickness;
		SerializedParameterOverride m_NormalSource;
		SerializedParameterOverride m_NormalCheckDirection;
		SerializedParameterOverride m_NormalMultiplier;
		SerializedParameterOverride m_NormalBias;
		SerializedParameterOverride m_NormalThreshold;

		public override void OnEnable()
		{
			m_DebugMode = FindParameterOverride(x => x.debugMode);
			m_OutlineThickness = FindParameterOverride(x => x.outlineThickness);
			m_OutlineColor = FindParameterOverride(x => x.outlineColor);

			m_EnableClosenessBoost = FindParameterOverride(x => x.enableClosenessBoost);
			m_ClosenessBoostThickness = FindParameterOverride(x => x.closenessBoostThickness);
			m_ClosenessBoostNear = FindParameterOverride(x => x.closenessBoostNear);
			m_ClosenessBoostFar = FindParameterOverride(x => x.closenessBoostFar);
			m_EnableDistanceFade = FindParameterOverride(x => x.enableDistanceFade);
			m_DistanceFadeNear = FindParameterOverride(x => x.distanceFadeNear);
			m_DistanceFadeFar = FindParameterOverride(x => x.distanceFadeFar);
			m_Depth9TilesThreshold = FindParameterOverride(x => x.depth9TilesThreshold);
			m_Depth9TilesBottomFix = FindParameterOverride(x => x.depth9TilesBottomFix);
			
			m_DepthSampleType = FindParameterOverride(x => x.depthSampleType);
			m_DepthThickness = FindParameterOverride(x => x.depthThickness);
			m_DepthMultiplier = FindParameterOverride(x => x.depthMultiplier);
			m_DepthBias = FindParameterOverride(x => x.depthBias);
			m_DepthThreshold = FindParameterOverride(x => x.depthThreshold);

			m_EnableNormalOutline = FindParameterOverride(x => x.enableNormalOutline);
			m_NormalThickness = FindParameterOverride(x => x.normalThickness);
			m_NormalSource = FindParameterOverride(x => x.normalSource);
			m_NormalCheckDirection = FindParameterOverride(x => x.normalCheckDirection);
			m_NormalMultiplier = FindParameterOverride(x => x.normalMultiplier);
			m_NormalBias = FindParameterOverride(x => x.normalBias);
			m_NormalThreshold = FindParameterOverride(x => x.normalThreshold);
		}

		public override void OnInspectorGUI()
		{
			EditorUtilities.DrawHeaderLabel("General");
			PropertyField(m_OutlineColor);
			PropertyField(m_OutlineThickness);

			EditorGUILayout.BeginVertical("box");
			PropertyField(m_EnableClosenessBoost);
			if (m_EnableClosenessBoost.value.boolValue)
            {
				PropertyField(m_ClosenessBoostThickness);
				PropertyField(m_ClosenessBoostNear);
				PropertyField(m_ClosenessBoostFar);
			}
			EditorGUILayout.EndVertical();

			EditorGUILayout.BeginVertical("box");
			PropertyField(m_EnableDistanceFade);
			if (m_EnableDistanceFade.value.boolValue)
            {
				PropertyField(m_DistanceFadeNear);
				PropertyField(m_DistanceFadeFar);
			}
			EditorGUILayout.EndVertical();

			EditorGUILayout.BeginVertical("box");
			EditorUtilities.DrawHeaderLabel("Depth");
			PropertyField(m_DepthThickness);
			PropertyField(m_DepthMultiplier);
			PropertyField(m_DepthBias);
			PropertyField(m_DepthThreshold);
			PropertyField(m_DepthSampleType);
			if (m_DepthSampleType.value.enumValueIndex == 0)
            {

            }
			else if (m_DepthSampleType.value.enumValueIndex == 1)
            {
				PropertyField(m_Depth9TilesThreshold);
				PropertyField(m_Depth9TilesBottomFix);
			}
			EditorGUILayout.EndVertical();

			EditorGUILayout.BeginVertical("box");
			EditorUtilities.DrawHeaderLabel("Normal");
			PropertyField(m_EnableNormalOutline);
			PropertyField(m_NormalSource);
			PropertyField(m_NormalThickness);
			PropertyField(m_NormalCheckDirection);
			PropertyField(m_NormalMultiplier);
			PropertyField(m_NormalBias);
			PropertyField(m_NormalThreshold);
			EditorGUILayout.EndVertical();

			EditorUtilities.DrawHeaderLabel("Debug");
			PropertyField(m_DebugMode);
		}
	}
}
