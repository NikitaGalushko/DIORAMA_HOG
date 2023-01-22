Shader "CR/CleanOutline"
{
	SubShader
	{
		Cull Off ZWrite Off ZTest Always
        Pass
		{
		 	CGPROGRAM

            #pragma vertex VertMain
            //#pragma fragment FragMainNormal
            #pragma fragment FragMain

            #include "UnityCG.cginc"

            sampler2D _MainTex;
            sampler2D _CameraDepthNormalsTexture;
            sampler2D _CameraDepthTexture;            
            sampler2D _CameraGBufferTexture0;
            sampler2D _CameraGBufferTexture2;

            float _OutlineThickness;
            float4 _OutlineColor;
            float _EnableClosenessBoost;
            float _ClosenessBoostThickness;
            float _BoostNear;
            float _BoostFar;

            float _EnableDistantFade;
            float _FadeNear;
            float _FadeFar;

            float _DepthCheckMoreSample;
            float _NineTilesThreshold;
            float _NineTileBottomFix;
            float _DepthThickness;
            float _OutlineDepthMultiplier;
            float _OutlineDepthBias;
            float _DepthThreshold;

            float _EnableNormalOutline;
            float _UseGBufferAsNormal;
            float _NormalCheckDirection;
            float _NormalThickness;
            float _OutlineNormalMultiplier;
            float _OutlineNormalBias;
            float _NormalThreshold;

            float _DebugMode; //0, off; 1, depth; 2, normal; 3, both
               
            struct VertData
            {
                float4 vertex : POSITION;
                float4 uv     : TEXCOORD0;
            };

            struct FragData
            {
                float4 vertex   : SV_POSITION;
                float2 texcoord : TEXCOORD0;
                float4 viewDir : TEXCOORD1;
            };

            FragData VertMain(VertData input)
            {
                FragData output;

                output.vertex = float4(input.vertex.xy, 0.0, 1.0);
                output.texcoord = (input.vertex.xy + 1.0) * 0.5;
	            #if UNITY_UV_STARTS_AT_TOP
	                output.texcoord = output.texcoord * float2(1.0, -1.0) + float2(0.0, 1.0);
	            #endif
				output.viewDir = mul (unity_CameraInvProjection, float4 (output.texcoord * 2.0 - 1.0, 1.0, 1.0));

                return output;
            }

            #ifndef MAX2
            #define MAX2(v) max(v.x, v.y)
            #endif
            #ifndef MIN2
            #define MIN2(v) min(v.x, v.y)
            #endif
            #ifndef MAX3
            #define MAX3(v) max(v.x, max(v.y, v.z))
            #endif
            #ifndef MIN3
            #define MIN3(v) min(v.x, min(v.y, v.z))
            #endif
            #ifndef MAX4
            #define MAX4(v) max(v.x, max(v.y, max(v.z, v.w)))
            #endif
            #ifndef MIN4
            #define MIN4(v) min(v.x, min(v.y, min(v.z, v.w)))
            #endif

            float remap(float value, float inputMin, float inputMax, float outputMin, float outputMax)
            {
                return (value - inputMin) * ((outputMax - outputMin) / (inputMax - inputMin)) + outputMin;
            }

            float3 GetNormal(sampler2D t, float2 uv)
            {
                float4 pixel = tex2D(t, uv);
                float depth;
                float3 normal;
                DecodeDepthNormal(pixel, depth, normal);
                float3 worldNormal = mul((float3x3)unity_MatrixInvV, float4(normal, 0.0));
                return worldNormal;
            }

            float DirectionalSampleNormal(sampler2D t, float2 uv, float3 offset)
            {
                float3 n_c = GetNormal(t, uv);
                float3 n_l = GetNormal(t, uv - offset.xz);
                float3 n_r = GetNormal(t, uv + offset.xz);
                float3 n_u = GetNormal(t, uv + offset.zy);
                float3 n_d = GetNormal(t, uv - offset.zy);

                float sobelNormalDot =  dot(n_l, n_c) + 
                                        dot(n_r, n_c) +
                                        dot(n_u, n_c) +
                                        dot(n_d, n_c);
                sobelNormalDot = remap(sobelNormalDot, -4, 4, 0, 1);
                sobelNormalDot = 1 - sobelNormalDot;

                return sobelNormalDot;
            }

            float3 SobelSampleGBufferNormal(float2 uv, float3 offset)
            {
                float3 n_c = GetNormal(_CameraGBufferTexture2, uv);
                float3 n_l   = GetNormal(_CameraGBufferTexture2, uv - offset.xz);
                float3 n_r  = GetNormal(_CameraGBufferTexture2, uv + offset.xz);
                float3 n_u     = GetNormal(_CameraGBufferTexture2, uv + offset.zy);
                float3 n_d   = GetNormal(_CameraGBufferTexture2, uv - offset.zy);

                return (n_c - n_l)  +
                       (n_c - n_r) +
                       (n_c - n_u)    +
                       (n_c - n_d);
            }

            float3 SobelSampleNormal(float2 uv, float3 offset)
            {
                if (_UseGBufferAsNormal == 1)
                {
                    return SobelSampleGBufferNormal(uv, offset);
                }

                float3 n_c = GetNormal(_CameraDepthNormalsTexture, uv);
                float3 n_l = GetNormal(_CameraDepthNormalsTexture, uv - offset.xz);
                float3 n_r = GetNormal(_CameraDepthNormalsTexture, uv + offset.xz);
                float3 n_u = GetNormal(_CameraDepthNormalsTexture, uv + offset.zy);
                float3 n_d = GetNormal(_CameraDepthNormalsTexture, uv - offset.zy);

                return (n_c - n_l) +
                       (n_c - n_r) +
                       (n_c - n_u) +
                       (n_c - n_d);
            }            

            //////////////////////////////////////////////////////////////////////////////////////
            //This MeshEdges function is from repository of Alexander Federwisch
            //https://github.com/Daodan317081/reshade-shaders
            ///BSD 3-Clause License
            // Copyright (c) 2018-2019, Alexander Federwisch
            // All rights reserved.
             float MeshEdges(float depthC, float4 depth1, float4 depth2) 
             {
                /******************************************************************************
                    Outlines type 2:
                    This method calculates how flat the plane around the center pixel is.
                    Can be used to draw the polygon edges of a mesh and its outline.
                ******************************************************************************/
                float depthCenter = depthC;
                float4 depthCardinal = float4(depth1.x, depth2.x, depth1.z, depth2.z);
                float4 depthInterCardinal = float4(depth1.y, depth2.y, depth1.w, depth2.w);
                //Calculate the min and max depths
                float2 mind = float2(MIN4(depthCardinal), MIN4(depthInterCardinal));
                float2 maxd = float2(MAX4(depthCardinal), MAX4(depthInterCardinal));
                float span = MAX2(maxd) - MIN2(mind) + 0.00001;

                //Normalize values
                depthCenter /= span;
                depthCardinal /= span;
                depthInterCardinal /= span;
                //Calculate the (depth-wise) distance of the surrounding pixels to the center
                float4 diffsCardinal = abs(depthCardinal - depthCenter);
                float4 diffsInterCardinal = abs(depthInterCardinal - depthCenter);
                //Calculate the difference of the (opposing) distances
                float2 meshEdge = float2(
                    max(abs(diffsCardinal.x - diffsCardinal.y), abs(diffsCardinal.z - diffsCardinal.w)),
                    max(abs(diffsInterCardinal.x - diffsInterCardinal.y), abs(diffsInterCardinal.z - diffsInterCardinal.w))
                );

                return MAX2(meshEdge);
            }
            /////////////////////////////////////////////////////////////////////////////////////


            float GetClosenessBoost(float3 viewPos, float depth01)
            {
                viewPos = viewPos * depth01;
                float dis = length (viewPos) * 0.01;
                float disBoost = smoothstep(_BoostFar, _BoostNear, dis) * _ClosenessBoostThickness + 1;
                return disBoost;
            }

            float GetDistanceFade(float3 viewPos, float depth01)
            {
                viewPos = viewPos * depth01;
                float dis = length (viewPos) * 0.01;
                float disBoost = smoothstep(_FadeFar, _FadeNear, dis) + 0.0001;
                return disBoost;
            }

            float MinFloats(float a, float b, float c, float d)
            {
                return min(min(a, b), min(c, d));
            }

            float SampleDepth9Tiles(sampler2D t, float2 uv, float3 offset, float3 viewPos, float centerDisBoost, float centerDepth, out float distanceFade)
            {
                offset *= centerDisBoost;

                float d_c = centerDepth;
                float d_l = tex2D(t, saturate(uv - offset.xz)).r;
                float d_r = tex2D(t, saturate(uv + offset.xz)).r;
                float d_u = tex2D(t, saturate(uv + offset.zy)).r;
                float d_d = tex2D(t, saturate(uv - offset.zy)).r;
                float d_lu = tex2D(t, saturate(uv + offset.xy * float2(-1,  1))).r;
                float d_ld = tex2D(t, saturate(uv + offset.xz * float2(-1, -1))).r;
                float d_ru = tex2D(t, saturate(uv + offset.xy * float2( 1,  1))).r;
                float d_rd = tex2D(t, saturate(uv + offset.xy * float2( 1, -1))).r;

                float d01_c = Linear01Depth(d_c);
                float d01_l = Linear01Depth(d_l);
                float d01_r = Linear01Depth(d_r);
                float d01_u = Linear01Depth(d_u);
                float d01_d = Linear01Depth(d_d); 
                float d01_lu = Linear01Depth(d_lu);
                float d01_ld = Linear01Depth(d_ld);
                float d01_ru = Linear01Depth(d_ru);
                float d01_rd = Linear01Depth(d_rd);

                float de_c = LinearEyeDepth(d_c);
                float de_l = LinearEyeDepth(d_l);
                float de_r = LinearEyeDepth(d_r);
                float de_u = LinearEyeDepth(d_u);
                float de_d = LinearEyeDepth(d_d);
                float de_lu = LinearEyeDepth(d_lu);
                float de_ld = LinearEyeDepth(d_ld);
                float de_ru = LinearEyeDepth(d_ru);
                float de_rd = LinearEyeDepth(d_rd);

                float depthC = de_c;
                float4 depth1 = float4(de_u, de_ru, de_r, de_rd);
                float4 depth2 = float4(de_d, de_ld, de_l, de_lu);
               
                distanceFade = 1;
                if (_EnableDistantFade == 1)
                {  
                     //get the smallest(closest) depth01 arround the center pixel
                    float closeDepth01 = min(MinFloats(d01_l,  d01_r,  d01_u,  d01_d ),
                                             MinFloats(d01_lu, d01_ld, d01_ru, d01_rd));
                    closeDepth01 = min(d01_c, closeDepth01);
                    distanceFade = GetDistanceFade(viewPos, closeDepth01);
                }

                float diff = MeshEdges(depthC, depth1, depth2);
                diff = smoothstep(_NineTilesThreshold, 1, diff) * distanceFade;

                float uvMask = smoothstep(0.0, _NineTileBottomFix, uv.y);
                diff *= uvMask;
                return diff;
            }

            float SampleDepth5Tiles(sampler2D t, float2 uv, float3 offset, float3 viewPos, float centerDisBoost, float centerDepth, out float distanceFade)
            {
                offset *= centerDisBoost;

                float d_c = centerDepth;
                float d_l = tex2D(t, uv - offset.xz).r;
                float d_r = tex2D(t, uv + offset.xz).r;
                float d_u = tex2D(t, uv + offset.zy).r;
                float d_d = tex2D(t, uv - offset.zy).r;

                float d01_c = Linear01Depth(d_c);
                float d01_l = Linear01Depth(d_l);
                float d01_r = Linear01Depth(d_r);
                float d01_u = Linear01Depth(d_u);
                float d01_d = Linear01Depth(d_d);

                float de_c = LinearEyeDepth(d_c);
                float de_l = LinearEyeDepth(d_l);
                float de_r = LinearEyeDepth(d_r);
                float de_u = LinearEyeDepth(d_u);
                float de_d = LinearEyeDepth(d_d);

                float diffSum = (de_c - de_l) + (de_c - de_r) + (de_c - de_u) + (de_c - de_d);

                distanceFade = 1;
                if (_EnableDistantFade == 1)
                {
                    //get the smallest(closest) depth01 arround the center pixel
                    float closeDepth01 = min(d01_c , MinFloats(d01_l, d01_r, d01_u, d01_d));
                    distanceFade = GetDistanceFade(viewPos, closeDepth01);
                }

                float result = abs(diffSum) * distanceFade;
                return result;
            }


            float4 FragMain(FragData input) : SV_Target
            {
                float2 uv = input.texcoord;
                float3 baseColor =  tex2D(_MainTex, uv).rgb;
                float3 finalColor = baseColor;

                float4 depth = tex2D(_CameraDepthTexture, input.texcoord.xy);
                float depth01 = Linear01Depth(depth.r);

                //will take consider screen resolution by default
                //you can modify these code to have more manual control
                float screenWidth = _ScreenParams.x;
                float screenHeight = _ScreenParams.y;
                float globalThickness = _OutlineThickness;
                globalThickness *= max(1, screenHeight / 1080);

                ////
                float3 offset = float3((1.0 / screenWidth), (1.0 / screenHeight), 0.0) * globalThickness;
                float3 viewPos = (input.viewDir.xyz / input.viewDir.w);

                float centerDisBoost = 1;
                if (_EnableClosenessBoost == 1)
                {
                    centerDisBoost = GetClosenessBoost(viewPos, depth01);
                }

                float sobelDepth = 0;
                float distanceFade = 1;
                if (_DepthCheckMoreSample == 1)
                {
                    sobelDepth =  SampleDepth9Tiles(_CameraDepthTexture, uv, offset * _DepthThickness, viewPos, centerDisBoost, depth.r, distanceFade);
                    sobelDepth = saturate(abs(sobelDepth));
                    sobelDepth = smoothstep(0, _DepthThreshold, sobelDepth) * sobelDepth;
                    sobelDepth = pow(sobelDepth* _OutlineDepthMultiplier, _OutlineDepthBias);
                }
                else
                {
                    sobelDepth = SampleDepth5Tiles(_CameraDepthTexture, uv, offset * _DepthThickness, viewPos, centerDisBoost, depth.r, distanceFade);
                    sobelDepth = saturate(abs(sobelDepth));
                    sobelDepth = smoothstep(0, _DepthThreshold, sobelDepth) * sobelDepth;
                    sobelDepth = pow(sobelDepth* _OutlineDepthMultiplier, _OutlineDepthBias);
                }

                float3 normalOffset = offset * _NormalThickness * centerDisBoost;
                float sobelNormal = 0;
                if (_EnableNormalOutline == 1)
                {
                    if (_NormalCheckDirection == 1)
                    {
                        if (_UseGBufferAsNormal == 1)
                        {
                            sobelNormal = DirectionalSampleNormal(_CameraGBufferTexture2, input.texcoord.xy, normalOffset);
                            sobelNormal = smoothstep(0, _NormalThreshold, sobelNormal) * sobelNormal;
                            sobelNormal = pow(abs(sobelNormal * _OutlineNormalMultiplier), _OutlineNormalBias);    
                        }
                        else
                        {
                            sobelNormal = DirectionalSampleNormal(_CameraDepthNormalsTexture, input.texcoord.xy, normalOffset);
                            sobelNormal = smoothstep(0, _NormalThreshold, sobelNormal) * sobelNormal;
                            sobelNormal = pow(abs(sobelNormal * _OutlineNormalMultiplier), _OutlineNormalBias);    
                        }
                        
                    }
                    else
                    {

                        float4 pixelCenter = tex2D(_CameraDepthNormalsTexture, uv);
                        float d_c;
                        float3 n_c;
                        DecodeDepthNormal(pixelCenter, d_c, n_c);

                        float3 sobelNormalVec = SobelSampleNormal(input.texcoord.xy, normalOffset);
                                                //abs(SobelSampleNormal(input.texcoord.xy, normalOffset));
                                    
                        sobelNormal = sqrt(dot(sobelNormalVec, sobelNormalVec));
                        sobelNormal = smoothstep(0, _NormalThreshold, sobelNormal) * sobelNormal;
                        sobelNormal = pow(abs(sobelNormal * _OutlineNormalMultiplier), _OutlineNormalBias);    

                    }
                }
               
                sobelNormal = saturate(abs(sobelNormal));
                sobelNormal *= distanceFade;

                float outlineStrength = max(sobelNormal, sobelDepth);


                if (_UseGBufferAsNormal == 1)
                {
                    float4 data = tex2D(_CameraGBufferTexture0, input.texcoord.xy);
                    float hasValidGBuffer = step(0.001, dot(data.xyz, data.xyz));
                    float isAtFarIstance = step(0.7, depth01);
                    outlineStrength = outlineStrength * max(hasValidGBuffer, isAtFarIstance);
                }

                float3 colorCombined = lerp(baseColor.rgb, outlineStrength * _OutlineColor, outlineStrength);

                if (_DebugMode == 0)
                {
                    finalColor.rgb = colorCombined;
                }
                else if (_DebugMode == 1)
                {
                    finalColor.rgb = sobelDepth;
                }
                else if (_DebugMode == 2)
                {
                    finalColor.rgb = sobelNormal;
                }
                else if (_DebugMode == 3)
                {
                    finalColor.rgb = outlineStrength;
                }
                return float4(finalColor.rgb, 1);
            }
            ENDCG
		}
	}
}