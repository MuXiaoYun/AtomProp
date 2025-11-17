import torch
from torch_geometric.data import Data, Batch
from atomprop.dataloader.dataloader import PyGChunkDataListLoader, xyzBatchLoaderContext
from atomprop.utils.groups import TripletGroup, QuadrupletGroup
import numpy as np

def save_1000_molecules_data(data_path, xyz_path, file_path):
    """
    保存前1000个分子的xyz坐标信息到指定文件
    """
    with xyzBatchLoaderContext(xyz_path) as xyz_loader:
        xyzs = xyz_loader.download_head(1000, file_path)
        print(f"已保存前1000个分子的XYZ坐标到 {file_path}")

def test_xyz_alignment_molecule_by_molecule(data_path, xyz_path, max_test_molecules=50):
    """
    逐个分子测试xyz坐标数据与分子数据的对齐情况，直到发现第一个不匹配
    """
    print("=== 开始逐个分子测试xyz坐标对齐 ===")
    
    # 创建数据加载器 - 使用小批量以便逐个检查
    total_rows = sum(1 for _ in open(data_path)) - 1
    test_size = min(max_test_molecules, total_rows)
    indices = np.arange(test_size)  # 测试前N个分子
    
    mismatch_found = False
    molecules_checked = 0
    
    with xyzBatchLoaderContext(xyz_path) as xyz_loader:
        data_loader = PyGChunkDataListLoader(
            data_path=data_path,
            split_indices=indices,
            chunk_size=1024,  # 较小的chunk_size
            max_atom_num=128,
            batch_size=16,    # 较小的batch_size
            file_type='txt'
        )
        
        total_xyz_atoms_so_far = 0
        
        for batch_idx, (data_list, mols) in enumerate(data_loader):
            if mismatch_found:
                break
                
            print(f"\n--- 处理批次 {batch_idx} ---")
            
            # 逐个分子检查
            for i, (data, mol) in enumerate(zip(data_list, mols)):
                molecules_checked += 1
                mol_atom_count = data.x.shape[0]
                
                print(f"\n🔍 检查分子 {molecules_checked}:")
                print(f"   SMILES: {mol}")
                print(f"   原子数量: {mol_atom_count}")
                print(f"   边数量: {data.edge_index.shape[1]}")
                
                # 计算当前分子在xyz数据中的起始和结束位置
                start_idx = total_xyz_atoms_so_far
                end_idx = total_xyz_atoms_so_far + mol_atom_count
                
                print(f"   在XYZ数据中的位置: [{start_idx}, {end_idx})")
                
                # 检查是否有足够的xyz数据
                try:
                    # 获取整个批次的xyz数据
                    batch_xyzs = xyz_loader.get_batch(len(data_list))
                    total_xyz_atoms = batch_xyzs.shape[0]
                    
                    print(f"   当前批次XYZ总原子数: {total_xyz_atoms}")
                    
                    if end_idx > total_xyz_atoms:
                        print(f"❌ 不匹配发现!")
                        print(f"   需要访问XYZ索引 [{start_idx}, {end_idx})")
                        print(f"   但XYZ数据只有 {total_xyz_atoms} 个原子")
                        print(f"   缺少 {end_idx - total_xyz_atoms} 个原子的坐标数据")
                        mismatch_found = True
                        break
                    
                    # 提取当前分子的xyz坐标
                    mol_xyzs = batch_xyzs[start_idx:end_idx]
                    
                    print(f"   ✅ XYZ数据足够")
                    print(f"   提取的坐标形状: {mol_xyzs.shape}")
                    
                    # 检查坐标合理性
                    if mol_xyzs.numel() > 0:
                        coord_stats = {
                            'min': mol_xyzs.min(dim=0)[0],
                            'max': mol_xyzs.max(dim=0)[0],
                            'mean': mol_xyzs.mean(dim=0),
                            'std': mol_xyzs.std(dim=0)
                        }
                        
                        print(f"   坐标统计:")
                        print(f"     X: [{coord_stats['min'][0]:.3f}, {coord_stats['max'][0]:.3f}] mean={coord_stats['mean'][0]:.3f}")
                        print(f"     Y: [{coord_stats['min'][1]:.3f}, {coord_stats['max'][1]:.3f}] mean={coord_stats['mean'][1]:.3f}")
                        print(f"     Z: [{coord_stats['min'][2]:.3f}, {coord_stats['max'][2]:.3f}] mean={coord_stats['mean'][2]:.3f}")
                        
                        # 检查是否有异常的坐标值
                        if (coord_stats['min'].abs() > 100).any() or (coord_stats['max'].abs() > 100).any():
                            print("   ⚠️  警告: 检测到可能异常的坐标值")
                    
                    # 测试三原子组生成和索引有效性
                    try:
                        triplet_indices = TripletGroup.batch_generate(data.edge_index)
                        print(f"   生成的三原子组数量: {triplet_indices.shape[0]}")
                        
                        if triplet_indices.shape[0] > 0:
                            # 检查三原子组索引是否在分子原子范围内
                            max_valid_idx = mol_atom_count - 1
                            invalid_triplets = (triplet_indices >= mol_atom_count).any(dim=1)
                            
                            if invalid_triplets.any():
                                print(f"❌ 三原子组索引错误!")
                                print(f"   分子原子索引范围: [0, {max_valid_idx}]")
                                print(f"   无效的三原子组: {triplet_indices[invalid_triplets]}")
                                mismatch_found = True
                                break
                            else:
                                print(f"   ✅ 三原子组索引有效")
                    except Exception as e:
                        print(f"   ❌ 三原子组生成错误: {e}")
                        mismatch_found = True
                        break
                        
                    # 测试四原子组生成
                    try:
                        quadruplet_indices = QuadrupletGroup.batch_generate(data.edge_index)
                        print(f"   生成的四原子组数量: {quadruplet_indices.shape[0]}")
                    except Exception as e:
                        print(f"   ❌ 四原子组生成错误: {e}")
                        mismatch_found = True
                        break
                    
                    # 更新累计原子计数
                    total_xyz_atoms_so_far += mol_atom_count
                    
                except Exception as e:
                    print(f"❌ 处理分子时发生错误: {e}")
                    mismatch_found = True
                    break
                
                print(f"   --- 分子 {molecules_checked} 检查完成 ---")
            
            if mismatch_found:
                break
        
        # 总结报告
        print(f"\n" + "="*50)
        print("测试总结:")
        print(f"检查的分子数量: {molecules_checked}")
        if mismatch_found:
            print(f"❌ 在第 {molecules_checked} 个分子发现不匹配!")
        else:
            print(f"✅ 所有检查的分子都匹配!")
        
        if molecules_checked > 0 and not mismatch_found:
            # 最终一致性检查
            try:
                final_batch_xyzs = xyz_loader.get_batch(1)  # 获取一个批次来检查总数
                total_expected_atoms = sum(data.x.shape[0] for data in data_list)  # 最后一个批次
                print(f"最终一致性检查:")
                print(f"  累计处理的原子数: {total_xyz_atoms_so_far}")
                print(f"  XYZ数据总原子数: {len(final_batch_xyzs)}")
                
                if total_xyz_atoms_so_far == len(final_batch_xyzs):
                    print("✅ 原子总数匹配!")
                else:
                    print(f"❌ 原子总数不匹配! 累计: {total_xyz_atoms_so_far}, XYZ总数: {len(final_batch_xyzs)}")
            except Exception as e:
                print(f"最终检查时出错: {e}")

def debug_xyz_loader_behavior(data_path, xyz_path, num_molecules=5):
    """
    调试xyz加载器的具体行为
    """
    print("\n" + "="*60)
    print("调试XYZ加载器行为")
    print("="*60)
    
    indices = np.arange(num_molecules)
    
    with xyzBatchLoaderContext(xyz_path) as xyz_loader:
        data_loader = PyGChunkDataListLoader(
            data_path=data_path,
            split_indices=indices,
            chunk_size=1024,
            max_atom_num=128,
            batch_size=2,  # 很小的批次以便观察
            file_type='txt'
        )
        
        for batch_idx, (data_list, mols) in enumerate(data_loader):
            print(f"\n--- 批次 {batch_idx} ---")
            print(f"请求 {len(data_list)} 个分子的XYZ数据")
            
            # 获取XYZ数据
            xyzs = xyz_loader.get_batch(len(data_list))
            print(f"获得的XYZ数据形状: {xyzs.shape}")
            
            # 显示每个分子的详细信息
            atom_counter = 0
            for i, (data, mol) in enumerate(zip(data_list, mols)):
                mol_atoms = data.x.shape[0]
                print(f"  分子 {i}: '{mol}' -> {mol_atoms} 个原子")
                print(f"    在XYZ数据中的位置: [{atom_counter}, {atom_counter + mol_atoms})")
                
                # 显示前3个原子的坐标（如果可用）
                if atom_counter + 3 <= xyzs.shape[0]:
                    sample_coords = xyzs[atom_counter:atom_counter+3]
                    print(f"    前3个原子坐标:")
                    for j, coord in enumerate(sample_coords):
                        print(f"      原子 {atom_counter+j}: ({coord[0]:.3f}, {coord[1]:.3f}, {coord[2]:.3f})")
                
                atom_counter += mol_atoms
            
            if batch_idx >= 2:  # 只查看前几个批次
                break

if __name__ == "__main__":
    data_path = "data/pubchem/pubchem-10m.txt"
    xyz_path = "data/pubchem/pubchem-xyzs.txt"

    print("测试配置:")
    print(f"数据路径: {data_path}")
    print(f"XYZ路径: {xyz_path}")
    
    # 保存前1000个分子的xyz数据
    save_1000_molecules_data(data_path, xyz_path, "pubchem-xyzs-head-1000.txt")

    # 运行主要测试
    test_xyz_alignment_molecule_by_molecule(data_path, xyz_path, max_test_molecules=50)
    
    # 运行调试测试
    debug_xyz_loader_behavior(data_path, xyz_path, num_molecules=5)