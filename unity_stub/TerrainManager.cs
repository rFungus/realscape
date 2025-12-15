using UnityEngine;

public class TerrainManager : MonoBehaviour
{
    public Terrain Terrain;

    public void ApplyHeightmap(float[,] heights)
    {
        if (Terrain == null || heights == null)
            return;

        int w = heights.GetLength(0);
        int h = heights.GetLength(1);
        Terrain.terrainData.heightmapResolution = Mathf.Max(w, h);
        Terrain.terrainData.SetHeights(0, 0, heights);
    }
}


