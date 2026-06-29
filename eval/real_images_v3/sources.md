# v3 Real Lunar Surface Image Sources

Staging directory: `v3-real-images/`
Curated: 2026-06-27
Total valid images: 21 (19 Apollo + 2 Surveyor 7 + 1 Surveyor 1 from NASA CDN)

All NASA photographs are works of the U.S. federal government created in the course of official duties and are therefore not subject to copyright protection under 17 U.S.C. § 105. They are in the public domain. Reference: https://www.nasa.gov/nasa-brand-center/images-and-media/

---

## Viewpoint notes

Task viewpoint: rocky regolith terrain, horizon, black sky — rover/lander/astronaut's-eye view.
Model segments three classes: {regolith, rock, sky}.

Images are tagged:
- [PANORAMA] = wide angle, horizon + sky likely visible — highest value for sim-to-real
- [TERRAIN] = ground-level with boulders or surface, horizon may or may not be visible
- [CLOSE-UP] = primarily a rock or regolith sample, limited or no sky
- [ASTRONAUT] = human figure visible in frame (not necessarily dominant)
- [EQUIPMENT] = lander/rover/hardware visible

---

## Chang'e / Yutu (CNSA) — SKIPPED

CNSA (China National Space Administration) does not publish imagery under any standard open license. Unlike NASA, CNSA has issued no blanket public-domain declaration for its mission photography. Lunar surface imagery from Chang'e 3/4/5 and Yutu-1/2 rovers circulates through Chinese state media, but reuse terms for non-Chinese entities are not specified. Licensing is ambiguous. Per the task brief's instruction ("include only if clearly permitted; if uncertain, SKIP"), **no CNSA imagery is included.**

---

## Apollo Mission Photography

All Apollo surface Hasselblad photographs were taken by astronauts in their official capacity as U.S. government employees and are public domain. NASA's usage policy: https://www.nasa.gov/nasa-brand-center/images-and-media/

Primary download source: NASA Images and Video Library CDN (`https://images-assets.nasa.gov/image/`)

---

### Apollo 11 — Mare Tranquillitatis (July 20, 1969)

---

#### 1. apollo11_tranquility_base_wide_panorama.jpg [PANORAMA]

- **NASA ID:** jsc2008e040725
- **Source URL:** https://images-assets.nasa.gov/image/jsc2008e040725/jsc2008e040725~large.jpg
- **Detail page:** https://images.nasa.gov/details/jsc2008e040725
- **Mission/instrument:** Apollo 11 / Hasselblad 500EL, 70mm surface camera (frames AS11-40-5954 through AS11-40-5961)
- **Description:** Panoramic composite assembled by JSC from Neil Armstrong's photographs at Tranquility Base, looking toward a small crater Armstrong noted during LM descent. Sky darkened to lunar horizon consistent with astronauts' observation that stars were not visible. No astronaut in frame.
- **License:** Public domain (NASA government work, 17 U.S.C. § 105)
- **Notes:** Best wide-angle Apollo 11 ground-level view; horizon and black sky clearly visible.

---

#### 2. apollo11_lm_shadow_regolith_footprints.jpg [TERRAIN]

- **NASA ID:** as11-37-5505
- **Source URL:** https://images-assets.nasa.gov/image/as11-37-5505/as11-37-5505~large.jpg
- **Detail page:** https://images.nasa.gov/details/as11-37-5505
- **Mission/instrument:** Apollo 11 / Hasselblad 70mm surface camera, July 20, 1969
- **Description:** Shadow of the Apollo 11 Lunar Module silhouetted against the lunar surface. Fine detail of astronaut footprint impressions in the regolith is visible. No astronaut body in frame; LM shadow and footprint tracks dominate.
- **License:** Public domain (NASA government work, 17 U.S.C. § 105)

---

#### 3. apollo11_regolith_footprint_closeup.jpg [CLOSE-UP]

- **NASA ID:** 6901250
- **Source URL:** https://images-assets.nasa.gov/image/6901250/6901250~large.jpg
- **Detail page:** https://images.nasa.gov/details/6901250
- **Mission/instrument:** Apollo 11 / 70mm surface close-up camera, July 20, 1969
- **Description:** Close-up of an astronaut's bootprint in the lunar regolith, photographed during the first lunar surface EVA. Classic reference for regolith texture — fine-grained compacted material with crisp impression walls.
- **License:** Public domain (NASA government work, 17 U.S.C. § 105)
- **Notes:** No horizon/sky; high value for regolith texture classification.

---

### Apollo 12 — Oceanus Procellarum / Surveyor Crater (Nov 19–20, 1969)

---

#### 4. apollo12_surveyor_crater_panorama.jpg [PANORAMA] [ASTRONAUT] [EQUIPMENT]

- **NASA ID:** jsc2011e118358
- **Source URL:** https://images-assets.nasa.gov/image/jsc2011e118358/jsc2011e118358~large.jpg
- **Detail page:** https://images.nasa.gov/details/jsc2011e118358
- **Mission/instrument:** Apollo 12 / Hasselblad 70mm (frames AS12-46-6777 through AS12-46-6780), Nov 19, 1969
- **Description:** Panoramic composite from Apollo 12's first moonwalk, positioned just inside the rim of Surveyor Crater. Assembled from four frames with minimal processing (lens flare removal, sky darkened to horizon). Astronaut Alan Bean and TV equipment are visible in frame. Horizon clearly shown.
- **License:** Public domain (NASA government work, 17 U.S.C. § 105)
- **Caution:** Bean and TV camera hardware visible. Use selectively — the horizon and regolith portions are usable; mask Bean + equipment in annotations if needed.

---

### Apollo 14 — Fra Mauro Highlands (Feb 5–6, 1971)

---

#### 5. apollo14_cone_crater_boulder_field_overview.jpg [TERRAIN]

- **NASA ID:** as14-64-9118
- **Source URL:** https://images-assets.nasa.gov/image/as14-64-9118/as14-64-9118~large.jpg
- **Detail page:** https://images.nasa.gov/details/as14-64-9118
- **Mission/instrument:** Apollo 14 / Hasselblad 70mm, Feb 6, 1971 (EVA-2)
- **Description:** Overall view looking south across the boulder field on the flank of Cone Crater. Multiple boulders of various sizes scattered across regolith, horizon visible in background. No astronaut as main subject.
- **License:** Public domain (NASA government work, 17 U.S.C. § 105)

---

#### 6. apollo14_cone_crater_boulder_field_south.jpg [TERRAIN]

- **NASA ID:** as14-64-9103
- **Source URL:** https://images-assets.nasa.gov/image/as14-64-9103/as14-64-9103~large.jpg
- **Detail page:** https://images.nasa.gov/details/as14-64-9103
- **Mission/instrument:** Apollo 14 / Hasselblad 70mm, Feb 6, 1971 (EVA-2)
- **Description:** Boulder field on the flank of Cone Crater, viewed from east looking south. Astronaut shadow may appear at edge. Terrain-dominant frame; rocks of varying sizes across regolith surface.
- **License:** Public domain (NASA government work, 17 U.S.C. § 105)

---

#### 7. apollo14_cone_crater_boulder_group.jpg [TERRAIN]

- **NASA ID:** as14-68-9453
- **Source URL:** https://images-assets.nasa.gov/image/as14-68-9453/as14-68-9453~large.jpg
- **Detail page:** https://images.nasa.gov/details/as14-68-9453
- **Mission/instrument:** Apollo 14 / Hasselblad 70mm, Feb 6, 1971 (EVA-2)
- **Description:** Group of large boulders near the rim of Cone Crater. White and brown rock visible with sample collection tool providing scale. Close-range view of boulder cluster on Fra Mauro highland regolith.
- **License:** Public domain (NASA government work, 17 U.S.C. § 105)

---

#### 8. apollo14_cone_crater_boulder_close.jpg [CLOSE-UP]

- **NASA ID:** as14-64-9135
- **Source URL:** https://images-assets.nasa.gov/image/as14-64-9135/as14-64-9135~large.jpg
- **Detail page:** https://images.nasa.gov/details/as14-64-9135
- **Mission/instrument:** Apollo 14 / Hasselblad 70mm, Feb 6, 1971 (EVA-2)
- **Description:** Close-up view of an approximately five-foot boulder encountered during the second EVA near Cone Crater. Shows rock surface texture, fracture patterns, and surrounding fine-grained regolith.
- **License:** Public domain (NASA government work, 17 U.S.C. § 105)

---

### Apollo 15 — Hadley-Apennine (July 31 – Aug 2, 1971)

---

#### 9. apollo15_station2_panorama.jpg [PANORAMA] [ASTRONAUT] [EQUIPMENT]

- **NASA ID:** as15-85-11451
- **Source URL:** https://images-assets.nasa.gov/image/as15-85-11451/as15-85-11451~large.jpg
- **Detail page:** https://images.nasa.gov/details/as15-85-11451
- **Mission/instrument:** Apollo 15 / Hasselblad 70mm (James Irwin, photographer), July 31, 1971
- **Description:** View looking northward along Hadley Rille from Station 2, with astronaut David Scott conducting geology at the Lunar Roving Vehicle on St. George Crater's flank. The dramatic Apennine Front and rille are visible in the background. Scott and LRV appear in lower portion of frame.
- **License:** Public domain (NASA government work, 17 U.S.C. § 105)
- **Caution:** Scott and LRV visible in lower portion; terrain, massif, and rille dominate the upper three-quarters of the frame.

---

#### 10. apollo15_hadley_rille_boulder.jpg [TERRAIN] [ASTRONAUT] [EQUIPMENT]

- **NASA ID:** as15-85-11437
- **Source URL:** https://images-assets.nasa.gov/image/as15-85-11437/as15-85-11437~large.jpg
- **Detail page:** https://images.nasa.gov/details/as15-85-11437
- **Mission/instrument:** Apollo 15 / Hasselblad 70mm (James Irwin, photographer), Aug 1, 1971
- **Description:** Commander David Scott examines a boulder with tongs and gnomon on the slope of Hadley Delta during the first moonwalk, with LRV visible nearby. Slope and highland terrain context visible around the human figure.
- **License:** Public domain (NASA government work, 17 U.S.C. § 105)
- **Caution:** Scott and LRV in frame as secondary elements; boulder and slope are prominent.

---

### Apollo 16 — Descartes Highlands (Apr 21–23, 1972)

---

#### 11. apollo16_south_ray_boulder_close.jpg [CLOSE-UP]

- **NASA ID:** as16-107-17573
- **Source URL:** https://images-assets.nasa.gov/image/as16-107-17573/as16-107-17573~large.jpg
- **Detail page:** https://images.nasa.gov/details/as16-107-17573
- **Mission/instrument:** Apollo 16 / Hasselblad 70mm, Apr 22, 1972 (EVA-2)
- **Description:** Close-up of a roughly half-meter block found near South Ray Crater, rolled over moments before photography. The block had remained undisturbed since crater formation. No astronaut in main view; rock and surrounding regolith fill the frame.
- **License:** Public domain (NASA government work, 17 U.S.C. § 105)

---

#### 12. apollo16_descartes_highlands_terrain.jpg [TERRAIN] [ASTRONAUT]

- **NASA ID:** as16-116-18671
- **Source URL:** https://images-assets.nasa.gov/image/as16-116-18671/as16-116-18671~large.jpg
- **Detail page:** https://images.nasa.gov/details/as16-116-18671
- **Mission/instrument:** Apollo 16 / Hasselblad 70mm (John Young, photographer), Apr 23, 1972 (EVA-3)
- **Description:** Charles Duke conducting geology at "Shadow Rock" near North Ray Crater. The rock earned its name from the permanently shadowed area it protects. Duke's figure is visible in the scene; the large rock dominates the upper frame with regolith surround.
- **License:** Public domain (NASA government work, 17 U.S.C. § 105)
- **Caution:** Astronaut Duke visible; Shadow Rock is the dominant subject.

---

#### 13. apollo16_shadow_rock_north_ray.jpg [TERRAIN]

- **NASA ID:** as16-117-18728
- **Source URL:** https://images-assets.nasa.gov/image/as16-117-18728/as16-117-18728~large.jpg
- **Detail page:** https://images.nasa.gov/details/as16-117-18728
- **Mission/instrument:** Apollo 16 / Hasselblad 70mm, Apr 23, 1972 (EVA-3, Station 13)
- **Description:** View of "Shadow Rock" southeast of North Ray Crater in the Descartes highlands. A geological scoop leans against the rock for scale. Rock surface and surrounding regolith fill the frame; no astronaut body in main view.
- **License:** Public domain (NASA government work, 17 U.S.C. § 105)

---

### Apollo 17 — Taurus-Littrow (Dec 11–14, 1972)

---

#### 14. apollo17_station6_boulder.jpg [TERRAIN]

- **NASA ID:** as17-137-20910
- **Source URL:** https://images-assets.nasa.gov/image/as17-137-20910/as17-137-20910~large.jpg
- **Detail page:** https://images.nasa.gov/details/as17-137-20910
- **Mission/instrument:** Apollo 17 / Hasselblad 70mm (Eugene Cernan or Harrison Schmitt), Dec 13, 1972 (EVA-2)
- **Description:** Large boulder at Station 6, Taurus-Littrow valley, photographed with Earth visible above it in the black lunar sky. One of the iconic Apollo 17 terrain shots — boulder, regolith, and Earth/sky all in frame. No astronaut body visible.
- **License:** Public domain (NASA government work, 17 U.S.C. § 105)
- **Notes:** Exceptional: shows rock, regolith, AND black sky (with Earth as bonus feature). High priority for test set.

---

#### 15. apollo17_station6_boulder_view.jpg [TERRAIN]

- **NASA ID:** as17-140-21438
- **Source URL:** https://images-assets.nasa.gov/image/as17-140-21438/as17-140-21438~large.jpg
- **Detail page:** https://images.nasa.gov/details/as17-140-21438
- **Mission/instrument:** Apollo 17 / Hasselblad 70mm, Dec 13, 1972 (EVA-3)
- **Description:** Large multi-cracked boulder at EVA Station 6, Taurus-Littrow. Samples were collected from this boulder during EVA-3. No astronaut as main subject; the boulder and surrounding regolith dominate the frame.
- **License:** Public domain (NASA government work, 17 U.S.C. § 105)

---

#### 16. apollo17_station6_split_rock_area.jpg [TERRAIN] [ASTRONAUT]

- **NASA ID:** as17-140-21496
- **Source URL:** https://images-assets.nasa.gov/image/as17-140-21496/as17-140-21496~large.jpg
- **Detail page:** https://images.nasa.gov/details/as17-140-21496
- **Mission/instrument:** Apollo 17 / Hasselblad 70mm (Eugene Cernan, photographer), Dec 13, 1972 (EVA-3)
- **Description:** Harrison Schmitt standing next to the massive split boulder at Station 6. The boulder is several meters tall; Schmitt provides scale. Lunar sky and South Massif terrain visible in background.
- **License:** Public domain (NASA government work, 17 U.S.C. § 105)
- **Caution:** Schmitt visible beside boulder; boulder and background terrain are dominant. Astronaut figure is relatively small compared to the boulder.

---

#### 17. apollo17_station5_camelot_panorama.jpg [TERRAIN]

- **NASA ID:** as17-145-22183
- **Source URL:** https://images-assets.nasa.gov/image/as17-145-22183/as17-145-22183~large.jpg
- **Detail page:** https://images.nasa.gov/details/as17-145-22183
- **Mission/instrument:** Apollo 17 / Hasselblad 70mm, Dec 12, 1972 (EVA-2, Station 5 / Camelot Crater)
- **Description:** Panoramic view looking northeast across the large boulder field at Station 5 near Camelot Crater. Multiple boulders distributed across regolith; Taurus-Littrow massifs visible in background. No astronaut in description; terrain dominant.
- **License:** Public domain (NASA government work, 17 U.S.C. § 105)

---

#### 18. apollo17_station2_terrain.jpg [CLOSE-UP]

- **NASA ID:** as17-137-20972
- **Source URL:** https://images-assets.nasa.gov/image/as17-137-20972/as17-137-20972~large.jpg
- **Detail page:** https://images.nasa.gov/details/as17-137-20972
- **Mission/instrument:** Apollo 17 / Hasselblad 70mm, Dec 12, 1972 (EVA-2, Station 2)
- **Description:** Close-up documentation of a lunar rock specimen with multi-colored mineral clasts embedded in a larger rock matrix. Station 2 sample documentation shot (SPL 2415, 2435-36, 40, 60). Geological tongs visible for scale.
- **License:** Public domain (NASA government work, 17 U.S.C. § 105)
- **Notes:** Close-up rock texture; limited sky. Useful for rock class testing.

---

## Surveyor Program Photography

Surveyor spacecraft were NASA programs; all imagery returned by their cameras is a product of U.S. government work and is in the public domain (17 U.S.C. § 105).

---

### Surveyor 1 (June 1966 — Oceanus Procellarum)

---

#### 19. surveyor1_surface_shadow.jpg [PANORAMA]

- **NASA ID:** PIA02976
- **Source URL:** https://images-assets.nasa.gov/image/PIA02976/PIA02976~large.jpg
- **Detail page:** https://images.nasa.gov/details/PIA02976
- **Mission/instrument:** Surveyor 1 / vidicon TV camera, June 1966
- **Credit:** NASA/JPL
- **Description:** Surveyor 1's own shadow cast across the lunar surface in the late lunar afternoon. Horizon is visible at upper right; the black lunar sky is clearly present. Regolith surface texture and scattered rocks visible in the foreground. Resolution ~1 mm/pixel at surface.
- **License:** Public domain (NASA/JPL government work, 17 U.S.C. § 105)
- **Notes:** One of the few ground-level images returned by Surveyor's own camera; confirms soft regolith and horizon viewpoint. High priority.

---

### Surveyor 7 (January 1968 — Tycho Crater ejecta blanket)

---

#### 20. surveyor7_surface_mosaic.jpg [PANORAMA]

- **Source URL:** https://assets.science.nasa.gov/dynamicimage/assets/science/psd/lunar-science/2023/08/surveyor7_mosaic.jpg
- **Reference page:** https://science.nasa.gov/resource/lunar-surface-mosaic-from-surveyor-7/
- **Mission/instrument:** Surveyor 7 / vidicon TV camera, January 1968
- **Description:** Hand-assembled mosaic of TV frames from Surveyor 7 showing the terrain near Tycho crater ejecta blanket. Rocky highland terrain, numerous boulders, and wide ground-level horizon visible. Surveyor 7 returned 21,274 photographs; this mosaic represents the character of the site.
- **License:** Public domain (NASA government work, 17 U.S.C. § 105)
- **Notes:** Highland terrain (different from Apollo maria sites) — more heavily cratered and rockier. Excellent for testing rock/regolith/sky segmentation.

---

#### 21. surveyor7_terrain_ground_view.png [TERRAIN] [EQUIPMENT]

- **Source URL:** https://planetary.s3.amazonaws.com/web/assets/pictures/20141024_surveyor-soil-sample-smaller.png
- **Reference page:** https://www.planetary.org/space-images/lunar-surface-surrounding-surveyor7
- **Mission/instrument:** Surveyor 7 / vidicon TV camera, January 1968
- **Credit:** NASA (hosted by The Planetary Society; Planetary Society notes "NASA images are in the public domain")
- **Description:** Ground-level view of the lunar surface terrain near Surveyor 7's landing site, with part of the spacecraft frame visible at image edge. Rocky ejecta blanket terrain near Tycho; typical highland surface.
- **License:** Public domain (underlying NASA image, 17 U.S.C. § 105)
- **Caution:** Surveyor spacecraft hardware visible at edge of frame.

---

## Summary

| Source | Count | Notes |
|--------|-------|-------|
| Apollo 11 | 3 | panorama + LM shadow + footprint close-up |
| Apollo 12 | 1 | Surveyor Crater panorama (Bean visible) |
| Apollo 14 | 4 | Cone Crater boulder field + close-ups |
| Apollo 15 | 2 | Hadley Delta (Scott + LRV visible) |
| Apollo 16 | 3 | South Ray + Shadow Rock area |
| Apollo 17 | 5 | Station 5 + Station 6 boulders (Earth in sky) |
| Surveyor 1 | 1 | Ground-level + horizon, own-camera image |
| Surveyor 7 | 2 | Highland mosaic + terrain view |
| Chang'e/Yutu | 0 | SKIPPED — CNSA licensing ambiguous (no open license declaration) |
| **Total** | **21** | |

**PANORAMA-class images** (horizon + black sky clearly visible, highest value for sky-class testing):
- apollo11_tranquility_base_wide_panorama.jpg
- apollo12_surveyor_crater_panorama.jpg
- apollo15_station2_panorama.jpg
- apollo17_station6_boulder.jpg (Earth above boulder in sky)
- apollo17_station5_camelot_panorama.jpg
- surveyor1_surface_shadow.jpg
- surveyor7_surface_mosaic.jpg

**Images with astronaut visible** (flagged; still useful if terrain is dominant area):
- apollo12_surveyor_crater_panorama.jpg
- apollo15_station2_panorama.jpg
- apollo15_hadley_rille_boulder.jpg
- apollo16_descartes_highlands_terrain.jpg
- apollo17_station6_split_rock_area.jpg
