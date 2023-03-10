using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Rendering.PostProcessing;
using System;

namespace CR
{

    public enum CleanOutlineDebugMode
    {
        Off,
        Depth,
        Normal,
        DepthAndNormal
    }

    public enum CleanOutlineDepthSample
    {
        FiveTiles,
        NineTiles
    }

    public enum CleanOutlineNormalSource
    {
        DepthNormalTexture,
        Gbuffer //deferred only
    }

    [Serializable]
    public sealed class CleanOutlineDebugModeParameter : ParameterOverride<CleanOutlineDebugMode> { }

    [Serializable]
    public sealed class CleanOutlineDepthSampleParameter : ParameterOverride<CleanOutlineDepthSample> { }

    [Serializable]
    public sealed class CleanOutlineNormalSourceParameter : ParameterOverride<CleanOutlineNormalSource> { }


    [PostProcess(typeof(CleanOutlineRenderer), PostProcessEvent.BeforeTransparent, "CR/CleanOutline")]
	public class CleanOutline : PostProcessEffectSettings
	{
        [Tooltip("Use debug mode to see outlines only")]
        public CleanOutlineDebugModeParameter debugMode = new CleanOutlineDebugModeParameter { value = CleanOutlineDebugMode.Off };

        [Tooltip("Color of outline")]
        public ColorParameter outlineColor = new ColorParameter { value = Color.black };

        [Tooltip("Thickness to both Depth and Normal outlines")]
        public FloatParameter outlineThickness = new FloatParameter { value = 1.0f };

        [Tooltip("At closer distance, the outline thickness will be boosted a little")]
        public BoolParameter enableClosenessBoost = new BoolParameter { value = false };
        [Range(0.01f, 5f)]
        [Tooltip("Extra boosted thickness at close range")]
        public FloatParameter closenessBoostThickness = new FloatParameter { value = 1 };
        [Range(0.00f, 2f)]
        [Tooltip("The near depth where boost will be 1")]
        public FloatParameter closenessBoostNear = new FloatParameter { value = 0.3f };
        [Range(0.01f, 2f)]
        [Tooltip("The far depth where boost will be 0")]
        public FloatParameter closenessBoostFar = new FloatParameter { value = 0.7f };

        [Tooltip("At further distance, the outline strength will be reduced, 0 at the farthest")]
        public BoolParameter enableDistanceFade = new BoolParameter { value = true };
        [Range(0.00f, 2f)]
        [Tooltip("The near depth where fade started as 1")]
        public FloatParameter distanceFadeNear = new FloatParameter { value = 0.15f };
        [Range(0.01f, 2f)]
        [Tooltip("The near depth where fade will reach 0 to give no outline")]
        public FloatParameter distanceFadeFar = new FloatParameter { value = 0.6f };
                
        ///depth
        [Tooltip("Depth Sample Type")]
        public CleanOutlineDepthSampleParameter depthSampleType = new CleanOutlineDepthSampleParameter { value = CleanOutlineDepthSample.FiveTiles };
        [Tooltip("Thickness of depth outline")]
        public FloatParameter depthThickness = new FloatParameter { value = 1.0f };
        [Tooltip("Multiplier of the depth outline strength")]
        public FloatParameter depthMultiplier = new FloatParameter { value = 1.0f };
        [Tooltip("Bias of the depth outline strength")]
        public FloatParameter depthBias = new FloatParameter { value = 1.0f };

        [Range(0f, 1f)]
        [Tooltip("Threshold of the depth outline, sample result lower than threshold will be ignored")]
        public FloatParameter depthThreshold = new FloatParameter { value = 0.1f };

        [Tooltip("Threshold to remap the NineTiles depth result")]
        [Range(0f, 1f)]
        public FloatParameter depth9TilesThreshold = new FloatParameter { value = 0.5f };
        [Range(0.0f, 0.1f)]
        [Tooltip("A bottom height to fix a weird effect in NineTiles sample")]
        public FloatParameter depth9TilesBottomFix = new FloatParameter { value = 0.005f };

        ///normal
        [Tooltip("Which texture will be use for normal, GBuffer only works in deferred pipeline")]
        public CleanOutlineNormalSourceParameter normalSource = new CleanOutlineNormalSourceParameter { value = CleanOutlineNormalSource.DepthNormalTexture };

        [Tooltip("Thickness of Normal")]
        public FloatParameter normalThickness = new FloatParameter { value = 1.0f };

        [Tooltip("Noraml check direction instead raw values, directions will be calculated into angles to check the difference")]
        public BoolParameter normalCheckDirection = new BoolParameter { value = true };

        [Tooltip("Multiplier of the normal outline strength")]
        public FloatParameter normalMultiplier = new FloatParameter { value = 5.0f };

        [Tooltip("Bias of the normal outline strength")]
        public FloatParameter normalBias = new FloatParameter { value = 10.0f };

        [Range(0f, 1f)]
        [Tooltip("Threshold of the normal outline, sample result lower than threshold will be ignored")]
        public FloatParameter normalThreshold = new FloatParameter { value = 0.3f };

        [Tooltip("Best practice is to normal outline only for lowploy scense, if for high poly and normal texture object it will appear weirdly")]
        public BoolParameter enableNormalOutline = new BoolParameter { value = true };
    }
    public class CleanOutlineRenderer : PostProcessEffectRenderer<CleanOutline>
    {
        public const string SHADER = "CR/CleanOutline";

        private int OUTLINETHICKNESS_ID = Shader.PropertyToID("_OutlineThickness");
        private int OUTLINECOLOR_ID = Shader.PropertyToID("_OutlineColor");
        private int ENABLECLOSENESSBOOST_ID = Shader.PropertyToID("_EnableClosenessBoost");
        private int CLOSENESSBOOSTTHICKNESS_ID = Shader.PropertyToID("_ClosenessBoostThickness");
        private int BOOSTNEAR_ID = Shader.PropertyToID("_BoostNear");
        private int BOOSTFAR_ID = Shader.PropertyToID("_BoostFar");
        private int ENABLEDISTANTFADE_ID = Shader.PropertyToID("_EnableDistantFade");
        private int FADENEAR_ID = Shader.PropertyToID("_FadeNear");
        private int FADEFAR_ID = Shader.PropertyToID("_FadeFar");
        private int DEPTHCHECKMORESAMPLE_ID = Shader.PropertyToID("_DepthCheckMoreSample");
        private int NINETILESTHRESHOLD_ID = Shader.PropertyToID("_NineTilesThreshold");
        private int NINETILEBOTTOMFIX_ID = Shader.PropertyToID("_NineTileBottomFix");
        private int DEPTHTHICKNESS_ID = Shader.PropertyToID("_DepthThickness");
        private int OUTLINEDEPTHMULTIPLIER_ID = Shader.PropertyToID("_OutlineDepthMultiplier");
        private int OUTLINEDEPTHBIAS_ID = Shader.PropertyToID("_OutlineDepthBias");
        private int DEPTHTHRESHOLD_ID = Shader.PropertyToID("_DepthThreshold");
        private int ENABLENORMALOUTLINE_ID = Shader.PropertyToID("_EnableNormalOutline");
        private int USEGBUFFERASNORMAL_ID = Shader.PropertyToID("_UseGBufferAsNormal");
        private int NORMALCHECKDIRECTION_ID = Shader.PropertyToID("_NormalCheckDirection");
        private int NORMALTHICKNESS_ID = Shader.PropertyToID("_NormalThickness");
        private int OUTLINENORMALMULTIPLIER_ID = Shader.PropertyToID("_OutlineNormalMultiplier");
        private int OUTLINENORMALBIAS_ID = Shader.PropertyToID("_OutlineNormalBias");
        private int NORMALTHRESHOLD_ID = Shader.PropertyToID("_NormalThreshold");
        private int DEBUGMODE_ID = Shader.PropertyToID("_DebugMode");


        public override void Render(PostProcessRenderContext context)
        {
            var shader = Shader.Find(SHADER);

            if (shader == null)
            {
                return;
            }
            var sheet = context.propertySheets.Get(shader);

            if (sheet == null)
            {
                return;
            }

            if (settings.normalSource.value == CleanOutlineNormalSource.DepthNormalTexture)
            {
                context.camera.depthTextureMode = context.camera.depthTextureMode | DepthTextureMode.DepthNormals;
            }

            //general
            sheet.properties.SetFloat(OUTLINETHICKNESS_ID, settings.outlineThickness);
            sheet.properties.SetColor(OUTLINECOLOR_ID, settings.outlineColor);
            

            sheet.properties.SetFloat(ENABLECLOSENESSBOOST_ID, settings.enableClosenessBoost ? 1 : 0);
            sheet.properties.SetFloat(CLOSENESSBOOSTTHICKNESS_ID, settings.closenessBoostThickness);
            sheet.properties.SetFloat(BOOSTNEAR_ID, settings.closenessBoostNear);
            sheet.properties.SetFloat(BOOSTFAR_ID, settings.closenessBoostFar);

            sheet.properties.SetFloat(ENABLEDISTANTFADE_ID, settings.enableDistanceFade ? 1 : 0);
            sheet.properties.SetFloat(FADENEAR_ID, settings.distanceFadeNear);
            sheet.properties.SetFloat(FADEFAR_ID, settings.distanceFadeFar);

            //depth
            sheet.properties.SetFloat(DEPTHCHECKMORESAMPLE_ID, (settings.depthSampleType.value == CleanOutlineDepthSample.NineTiles) ? 1 : 0);
            sheet.properties.SetFloat(DEPTHTHICKNESS_ID, settings.depthThickness);

            sheet.properties.SetFloat(DEPTHTHRESHOLD_ID, settings.depthThreshold);

            sheet.properties.SetFloat(OUTLINEDEPTHMULTIPLIER_ID, settings.depthMultiplier);
            sheet.properties.SetFloat(OUTLINEDEPTHBIAS_ID, settings.depthBias);
            sheet.properties.SetFloat(NINETILESTHRESHOLD_ID, settings.depth9TilesThreshold);
            sheet.properties.SetFloat(NINETILEBOTTOMFIX_ID, settings.depth9TilesBottomFix);

            //normal
            sheet.properties.SetFloat(ENABLENORMALOUTLINE_ID, settings.enableNormalOutline ? 1 : 0);
            sheet.properties.SetFloat(NORMALTHICKNESS_ID, settings.normalThickness);
            sheet.properties.SetFloat(USEGBUFFERASNORMAL_ID, (float)settings.normalSource.value);
            sheet.properties.SetFloat(OUTLINENORMALMULTIPLIER_ID, settings.normalMultiplier);
            sheet.properties.SetFloat(OUTLINENORMALBIAS_ID, settings.normalBias);
            sheet.properties.SetFloat(NORMALTHRESHOLD_ID, settings.normalThreshold);
            sheet.properties.SetFloat(NORMALCHECKDIRECTION_ID, settings.normalCheckDirection ? 1 : 0);

            //debug
            sheet.properties.SetFloat(DEBUGMODE_ID, (float)settings.debugMode.value);

            context.command.BlitFullscreenTriangle(context.source, context.destination, sheet, 0);

        }
    }
}
