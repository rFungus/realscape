using System.Threading.Tasks;
using UnityEngine;

public class HeightmapProvider : MonoBehaviour
{
    public int Resolution = 64;

    public async Task<float[,]> GetHeightmapAsync()
    {
        // Заглушка: возвращает плоский ландшафт
        await Task.Yield();
        var map = new float[Resolution, Resolution];
        return map;
    }
}


