#version 440

layout(r32i, binding = 0) uniform iimage2D sortingTexture;
uniform float time;  // Controls animation speed
uniform int width, height, screenWidth, screenHeight;
uniform int maxValue;


out vec4 FragColor;

vec4 lerp(vec4 a, vec4 b, float t){
    return a + (b - a) * t;
}

float inverseLerp(float a, float b, float v){
    return (v - a) / (b - a);
}

// From https://gist.github.com/mairod/a75e7b44f68110e1576d77419d608786
vec3 hueShift( vec3 color, float hueAdjust ){

    const vec3  kRGBToYPrime = vec3 (0.299, 0.587, 0.114);
    const vec3  kRGBToI      = vec3 (0.596, -0.275, -0.321);
    const vec3  kRGBToQ      = vec3 (0.212, -0.523, 0.311);

    const vec3  kYIQToR     = vec3 (1.0, 0.956, 0.621);
    const vec3  kYIQToG     = vec3 (1.0, -0.272, -0.647);
    const vec3  kYIQToB     = vec3 (1.0, -1.107, 1.704);

    float   YPrime  = dot (color, kRGBToYPrime);
    float   I       = dot (color, kRGBToI);
    float   Q       = dot (color, kRGBToQ);
    float   hue     = atan (Q, I);
    float   chroma  = sqrt (I * I + Q * Q);

    hue += hueAdjust;

    Q = chroma * sin (hue);
    I = chroma * cos (hue);

    vec3    yIQ   = vec3 (YPrime, I, Q);

    return vec3( dot (yIQ, kYIQToR), dot (yIQ, kYIQToG), dot (yIQ, kYIQToB) );

}

void main() {
    float modTime = time;
    modTime -= 4;
    modTime = max(modTime, 0);
    modTime *= 8;

    vec2 pos = gl_FragCoord.xy / vec2(screenWidth, screenHeight);
    pos.y = 1 - pos.y;
    pos.x = 1 - pos.x;

    ivec2 texCoord = ivec2(pos.x * width, min(int(floor(modTime)), height - 1));

    int val = imageLoad(sortingTexture, texCoord).r;
    int prevVal = imageLoad(sortingTexture, ivec2(texCoord.x, texCoord.y - 1)).r;
    float normalizedVal = float(val) / maxValue;

    float highlightT = mod(modTime, 1);
    vec4 finCol = vec4(0.9, 0.9, 0.9, 1);
    vec4 startCol = vec4(0.9, 0.9, 0.9, 1);

    highlightT = highlightT * highlightT;

    if(texCoord.y - 1 >= 0 && prevVal != val && height / modTime > 1){
        startCol = vec4(0.980392157, 77/255, 57/255, 1);
    }

    if(pos.y < normalizedVal){
        FragColor = vec4(0,0,0,1);
    }else{
        float mul = 0.03 + inverseLerp(0, 1 - normalizedVal, 1 - pos.y) * 0.9;
        vec4 col = lerp(startCol, finCol, highlightT);
        FragColor = lerp(vec4(col.r * mul, 0, 0, 1), vec4(1,1,1,0), mul);
        FragColor = vec4(hueShift(FragColor.rgb, time * 0.8 + (1 - normalizedVal) * 2), 1);

        if(texCoord.y - 1 >= 0 && prevVal != val && height / modTime > 1){
            FragColor = lerp(col, FragColor, highlightT);
        }
    }
}
