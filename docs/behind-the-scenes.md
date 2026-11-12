[Jak 1](../README.md) · [Jak II](https://github.com/JacopoBassan/the-cinematic-legacy-jak2/blob/main/README.md) · [Jak 3](https://github.com/JacopoBassan/the-cinematic-legacy-jak3/blob/main/README.md)

# Behind the scenes

I [asked about this in August 2024](https://discord.com/channels/756287461377703987/949828064026714174/1268810551509909575). Two years later, I took matters into my own hands.

## First steps

![Early Jak 3 rig prototype in Blender](media/02-first-rig-prototype.webp)

*September 2026. An early Blender rig.*

![Early Jak 3 in-game prototype](media/01-early-jak3-gameplay.webp)

*In-game later that day. A few things were still off.*

## The source models

For Jak 1, I used the cinematic models from Jak II, including Jak’s classic look from the opening cutscene.

*Model views. Gameplay left, cinematic right.*

### Jak 1 · Jak

![Jak 1 gameplay and Jak II cinematic source meshes](media/jak1-jak-sources.webp)

Gameplay: **3,593 triangles** · Cinematic: **9,886 triangles**.

### Jak II · Ashelin

![Jak II gameplay and cinematic Ashelin source meshes](media/jak2-ashelin-sources.webp)

Gameplay: **3,295 triangles** · Cinematic: **10,223 triangles**.

### Jak 3 · Daxter

![Jak 3 gameplay and cinematic Daxter source meshes](media/jak3-daxter-sources.webp)

Gameplay: **1,296 triangles** · Cinematic: **9,510 triangles**.

## Faces and bones

The cinematic meshes needed to work with the gameplay skeletons.

![Gameplay skeleton and facial landmarks](media/rig-and-face.webp)

*Jak 1 model view. Bones and brow joints in teal; mouth reference points in gold.*

![Daxter face point cloud from actual model vertices](media/01-daxter-face-point-cloud.webp)

*Left: gameplay expression. Centre: neutral cinematic face. Right: fitted expression.*

I adapted each replacement to its game’s original rig and animation controls. The facial work included fitting cinematic blend shapes to gameplay expressions, plus separate work on eyes, jaws and brows. Daxter’s face above is one example.

## Things got weird

Early Windows builds from October 2026.

![Early Jak 3 skirt stretching into a giant ribbon](media/03-jak3-skirt-ribbon.webp)

*The skirt was a little long.*

![Keira’s trouser cuff stretched across the workshop](media/02-keira-runaway-cuff.webp)

*Keira’s trouser cuff stretched across the workshop.*

![Samos’s beard passing through his torso before the fix, then restored](media/samos-beard-before-after.webp)

*Samos’s beard went through his belt. Before and after the fix.*

## Getting details right

![Jak 1 mouth before and after fixing the mouth interior and teeth](media/05-jak1-mouth-before-after.webp)

*Jak 1, before and after fixing the mouth and teeth. November 2026.*

## Finished examples

Original on the left, cinematic on the right.

### Jak 1

![Jak 1 comparison after the mouth fix](comparisons/jak1/jak-front-ae65be5000ef.webp)

![Jak and Daxter in the final Naughty Dog intro pose, original and cinematic](media/jak1-ndi-logo-comparison.webp)

![Closer view of the original and cinematic character models in the final intro pose](media/jak1-ndi-characters-comparison.webp)

*The final pose in Jak 1’s Naughty Dog intro. Original left, cinematic right.*

### Jak II · Jak and Daxter

![Jak II Jak and Daxter comparison](media/jak2-hero.jpg)

### Jak II · Sig

![Jak II Sig comparison](media/jak2-sig.jpg)

### Jak II · Ashelin

![Jak II Ashelin comparison](media/jak2-ashelin.jpg)

### Jak 3 · Jak and Daxter

![Jak 3 Jak and Daxter comparison](media/jak3-hero.webp)

### Jak 3 · Daxter

![Jak 3 Daxter comparison](media/jak3-daxter.webp)

I’ve put a lot of care into checking these models, though full playthroughs still need testing. If I’ve missed anything, just let me know and I’ll get it fixed! I might already be working on a few little touch-ups here and there.
