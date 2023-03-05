
import torch
from d2l import torch as d2l
from matplotlib import pyplot as plt

torch.set_printoptions(2)

def multibox_prior(data, sizes, ratios):
    """生成以每个像素为中心具有不同形状的锚框"""
    in_height, in_width = data.shape[-2:]
    device, num_sizes, num_ratios = data.device, len(sizes), len(ratios)
    boxes_per_pixel = (num_sizes + num_ratios - 1)
    size_tensor = torch.tensor(sizes, device=device)
    ratio_tensor = torch.tensor(ratios, device=device)

    # 为了将锚点移动到像素的中心，需要设置偏移量。
    # 因为一个像素的高为1且宽为1，我们选择偏移我们的中心0.5
    offset_h, offset_w = 0.5, 0.5
    steps_h = 1.0 / in_height  # 在y轴上缩放步长
    steps_w = 1.0 / in_width  # 在x轴上缩放步长

    # 生成锚框的所有中心点
    #对应像素点个数的偏移位置中心点的比值
    center_h = (torch.arange(in_height, device=device) + offset_h) * steps_h
    center_w = (torch.arange(in_width, device=device) + offset_w) * steps_w
    shift_y, shift_x = torch.meshgrid(center_h, center_w, indexing='ij')
    shift_y, shift_x = shift_y.reshape(-1), shift_x.reshape(-1)
    #长度为h*w的从最小（h或w个）到最大一次排序的一维向量,组合后为所有中心点的坐标

    # 生成“boxes_per_pixel”个高和宽，
    # 之后用于创建锚框的四角坐标(xmin,xmax,ymin,ymax)(本质是和原图片宽的对应比值)
    w = torch.cat((size_tensor * torch.sqrt(ratio_tensor[0]),
                   sizes[0] * torch.sqrt(ratio_tensor[1:])))\
                   * in_height / in_width  # 处理矩形输入
    h = torch.cat((size_tensor / torch.sqrt(ratio_tensor[0]),
                   sizes[0] / torch.sqrt(ratio_tensor[1:])))
    # 除以2来获得半高和半宽
    # 其中返回的w，h为拼接的总的锚框种类（分别对于h和w上的缩放比例）
    #其中ratio——tensor[1：]特别重要，因为这样会少一个比值对产生，使得数量正常
    # 输出数据和中心点结构相同，w和h11对应且个数为（s+r-1）

    anchor_manipulations = torch.stack((-w, -h, w, h)).T.repeat(
                                        in_height * in_width, 1) / 2

    # 每个中心点都将有“boxes_per_pixel”个锚框
    # 所以生成含所有锚框中心的网格，重复了“boxes_per_pixel”次
    # 其中数据为和原图片w，h的缩放比例
    # 个数为 （s+r-1）*w*h 个

    out_grid = torch.stack([shift_x, shift_y, shift_x, shift_y],
                dim=1).repeat_interleave(boxes_per_pixel, dim=0)
    output = out_grid + anchor_manipulations
    return output.unsqueeze(0)
#输出格式中[批大小，锚框数量，位置左上右下坐标]

img = d2l.plt.imread('../pic/dog.png')
h, w = img.shape[:2]

print(h, w)
X = torch.rand(size=(1, 3, h, w))
Y = multibox_prior(X, sizes=[0.75, 0.5, 0.25], ratios=[1, 2, 0.5])
print(Y.shape)

boxes = Y.reshape(h, w, 5, 4)
print(boxes[250, 250, 0, :])

def show_bboxes(axes, bboxes, labels=None, colors=None):
    """显示所有边界框"""
    def _make_list(obj, default_values=None):
        if obj is None:
            obj = default_values
        elif not isinstance(obj, (list, tuple)):
            obj = [obj]
        return obj

    labels = _make_list(labels)
    colors = _make_list(colors, ['b', 'g', 'r', 'm', 'c'])
    for i, bbox in enumerate(bboxes):
        color = colors[i % len(colors)]
        rect = d2l.bbox_to_rect(bbox.detach().numpy(), color)
        if labels and len(labels) > i:
            text_color = 'k' if color == 'w' else 'w'
            axes.text(rect.xy[0], rect.xy[1], labels[i],
                      va='center', ha='center', fontsize=9, color=text_color,
                      bbox=dict(facecolor=color, lw=0))

d2l.set_figsize()
bbox_scale = torch.tensor((w, h, w, h))
fig = d2l.plt.imshow(img)
show_bboxes(fig.axes, boxes[250, 250, :, :] * bbox_scale,
            ['s=0.75, r=1', 's=0.5, r=1', 's=0.25, r=1', 's=0.75, r=2',
             's=0.75, r=0.5'])

def box_iou(boxes1, boxes2):
    """计算两个锚框或边界框列表中成对的交并比"""
    box_area = lambda boxes: ((boxes[:, 2] - boxes[:, 0]) *
                              (boxes[:, 3] - boxes[:, 1]))
    # boxes1,boxes2,areas1,areas2的形状:
    # boxes1：(boxes1的数量,4),
    # boxes2：(boxes2的数量,4),
    # areas1：(boxes1的数量,),
    # areas2：(boxes2的数量,)
    areas1 = box_area(boxes1)
    areas2 = box_area(boxes2)
    # inter_upperlefts,inter_lowerrights,inters的形状:
    # (boxes1的数量,boxes2的数量,2)
    inter_upperlefts = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    inter_lowerrights = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    #使用None进行升维,并且位置关键，其中会使其变成[2,1,2]，生成每个锚框（每个二维部分）和所有标示框（整体为二维部分）的交集
    #左上比较最大，右下比较最小
    inters = (inter_lowerrights - inter_upperlefts).clamp(min=0)
    # inter_areasandunion_areas的形状:(boxes1的数量,boxes2的数量)
    inter_areas = inters[:, :, 0] * inters[:, :, 1]
    union_areas = areas1[:, None] + areas2 - inter_areas
    return inter_areas / union_areas
    #在算法中，选取单个变量计算会导致降维，因此使用None进行升维(None的位置有影响)

def assign_anchor_to_bbox(ground_truth, anchors, device, iou_threshold=0.5):
    #(真实框，锚框，数据位置，抑制阈值（真实框的拟合最大值如果小于该值自动去除不看）)
    #注意，此算法中一个真实框可能对应多个锚框
    """将最接近的真实边界框分配给锚框"""
    num_anchors, num_gt_boxes = anchors.shape[0], ground_truth.shape[0]
    # 位于第i行和第j列的元素x_ij是锚框i和真实边界框j的IoU
    jaccard = box_iou(anchors, ground_truth)
    # 对于每个锚框，分配的真实边界框的张量
    #竖着的纵轴是锚框，横着的横轴是真实框
    anchors_bbox_map = torch.full((num_anchors,), -1, dtype=torch.long,
                                  device=device)
    # 根据阈值，决定是否分配真实边界框
    max_ious, indices = torch.max(jaccard, dim=1)
    # 横向最大和坐标,即获得每个真实框的最大iou对应的锚框位置
    anc_i = torch.nonzero(max_ious >= iou_threshold).reshape(-1)#坐标
    box_j = indices[max_ious >= iou_threshold]#值（本质是竖向位置，此时对应分配的真实框）
    #虽然anci与boxj的小于部分被抑制且size改变，但是位置任然一一对应（因为筛选条件一样）
    anchors_bbox_map[anc_i] = box_j
    #最初始的真实框分配，没有考虑重复
    col_discard = torch.full((num_anchors,), -1)
    row_discard = torch.full((num_gt_boxes,), -1)
    for _ in range(num_gt_boxes):
        max_idx = torch.argmax(jaccard)
        box_idx = (max_idx % num_gt_boxes).long()
        anc_idx = (max_idx / num_gt_boxes).long()
        anchors_bbox_map[anc_idx] = box_idx
        jaccard[:, box_idx] = col_discard
        jaccard[anc_idx, :] = row_discard
    #输出为一维向量，其中-1代表没有匹配，非负一值代表其匹配的真实框编号
    #[anchors]
    return anchors_bbox_map

def offset_boxes(anchors, assigned_bb, eps=1e-6):
    """对锚框偏移量的转换"""
    c_anc = d2l.box_corner_to_center(anchors)
    c_assigned_bb = d2l.box_corner_to_center(assigned_bb)
    offset_xy = 10 * (c_assigned_bb[:, :2] - c_anc[:, :2]) / c_anc[:, 2:]
    offset_wh = 5 * torch.log(eps + c_assigned_bb[:, 2:] / c_anc[:, 2:])
    offset = torch.cat([offset_xy, offset_wh], axis=1)
    #[anchors_num,4] and assigned
    return offset

def multibox_target(anchors, labels):
    """使用真实边界框标记锚框"""
    batch_size, anchors = labels.shape[0], anchors.squeeze(0)
    batch_offset, batch_mask, batch_class_labels = [], [], []
    device, num_anchors = anchors.device, anchors.shape[0]
    for i in range(batch_size):
        #每个图片的真实框和锚框不一样，每个batch单独算
        #第一个数据为种类，后四个数据为真实框
        label = labels[i, :, :]
        anchors_bbox_map = assign_anchor_to_bbox(
            label[:, 1:], anchors, device)
        #对保留的（匹配到真实框的锚框）进行标记，对位值为1有，0没有
        bbox_mask = ((anchors_bbox_map >= 0).float().unsqueeze(-1)).repeat(
            1, 4)#在结果大小下提前对非背景框进行标记
        # 将类标签和分配的边界框坐标初始化为零
        class_labels = torch.zeros(num_anchors, dtype=torch.long,
                                   device=device)
        assigned_bb = torch.zeros((num_anchors, 4), dtype=torch.float32,
                                  device=device)
        # 使用真实边界框来标记锚框的类别。
        # 如果一个锚框没有被分配，我们标记其为背景（值为零）
        indices_true = torch.nonzero(anchors_bbox_map >= 0)#有分配的锚框的坐标，在一维上
        bb_idx = anchors_bbox_map[indices_true]#有分配的锚框的真实类别值，同样一维度
        #当所以为矩阵是，以原位置，其中为对应元素输出
        #因为存在0类图片，而0在其中表示背景，所以+1
        #label:[anchors,1+4]
        #当进行赋值时，
        class_labels[indices_true] = label[bb_idx, 0].long() + 1#锚框对应图片种类（将真实框的种类赋给[anchors]上的对应锚框）
        assigned_bb[indices_true] = label[bb_idx, 1:]#锚框对应真实框值(将真实框的坐标赋给[anchors，4]上的对应锚框)，此时indices虽然为一维但是对应二维坐标
        # 偏移量转换
        offset = offset_boxes(anchors, assigned_bb) * bbox_mask#防止未分配的背景框获得偏移值（其实本身就为0只是预防）
        batch_offset.append(offset.reshape(-1))#[anchors*4]
        batch_mask.append(bbox_mask.reshape(-1))
        batch_class_labels.append(class_labels)
    bbox_offset = torch.stack(batch_offset)
    bbox_mask = torch.stack(batch_mask)
    class_labels = torch.stack(batch_class_labels)
    #[batch_size,anchors*4(offset)],[batch_size,anchors*4(sign)],[batch_size,anchors(对应种类+1)]
    return (bbox_offset, bbox_mask, class_labels)

ground_truth = torch.tensor([[0, 0.1, 0.08, 0.52, 0.92],
                         [1, 0.55, 0.2, 0.9, 0.88]])
anchors = torch.tensor([[0, 0.1, 0.2, 0.3], [0.15, 0.2, 0.4, 0.4],
                    [0.63, 0.05, 0.88, 0.98], [0.66, 0.45, 0.8, 0.8],
                    [0.57, 0.3, 0.92, 0.9]])

fig = d2l.plt.imshow(img)
show_bboxes(fig.axes, ground_truth[:, 1:] * bbox_scale, ['dog', 'cat'], 'k')
show_bboxes(fig.axes, anchors * bbox_scale, ['0', '1', '2', '3', '4'])
plt.show()

labels = multibox_target(anchors.unsqueeze(dim=0),
                         ground_truth.unsqueeze(dim=0))

print(labels[2])
print(labels[1])
print(labels[0])

def offset_inverse(anchors, offset_preds):
    """根据带有预测偏移量的锚框来预测边界框"""
    #根据锚框中心点和预测的偏移量来返回预测的锚框
    anc = d2l.box_corner_to_center(anchors)
    pred_bbox_xy = (offset_preds[:, :2] * anc[:, 2:] / 10) + anc[:, :2]
    pred_bbox_wh = torch.exp(offset_preds[:, 2:] / 5) * anc[:, 2:]
    pred_bbox = torch.cat((pred_bbox_xy, pred_bbox_wh), axis=1)
    predicted_bbox = d2l.box_center_to_corner(pred_bbox)
    return predicted_bbox

def nms(boxes, scores, iou_threshold):
    """对预测边界框的置信度进行排序"""
    B = torch.argsort(scores, dim=-1, descending=True)#返回最大值的索引，此时为一维向量，对应值为锚框标号
    keep = []  # 保留预测边界框的指标
    while B.numel() > 0:
        i = B[0]
        keep.append(i)
        if B.numel() == 1: break
        #计算最大的[0]地和剩下的所有的iou
        iou = box_iou(boxes[i, :].reshape(-1, 4),
                      boxes[B[1:], :].reshape(-1, 4)).reshape(-1)
        inds = torch.nonzero(iou <= iou_threshold).reshape(-1)#此时tensor的大小为anchors-1，因此对应的原坐标需要+1
        B = B[inds + 1]#切片，选出不相似的
    return torch.tensor(keep, device=boxes.device)#此时抑制阈值不得过小和过大，不然会导致一定程度的样本丢失或者抑制效果差
#[batch_size,classes_num+1,num_ach_total],   [batch_size,anchors(h*w)*4(offset)],     [batch_size,num_ach,4]
def multibox_detection(cls_probs, offset_preds, anchors, nms_threshold=0.5,
                       pos_threshold=0.009999999):
    """使用非极大值抑制来预测边界框"""
    device, batch_size = cls_probs.device, cls_probs.shape[0]
    anchors = anchors.squeeze(0)
    num_classes, num_anchors = cls_probs.shape[1], cls_probs.shape[2]#C这里的classes是比原来的+1的，包括没有（0）
    out = []
    for i in range(batch_size):
        #[classes_num+1,num_ach_total],[anchors(h*w),4(offset)]
        cls_prob, offset_pred = cls_probs[i], offset_preds[i].reshape(-1, 4)
        conf, class_id = torch.max(cls_prob[1:], 0)# 这里直接无视了预测不是（0）的概率，然后求最大，保证了传入的概率为一维向量[anchors_total]
        predicted_bb = offset_inverse(anchors, offset_pred)#返回预测的真实框坐标
        #真实框坐标，最大可能，抑制阈值（通过for保证每次只输入一张图片）
        keep = nms(predicted_bb, conf, nms_threshold)

        # 找到所有的non_keep索引，并将类设置为背景
        all_idx = torch.arange(num_anchors, dtype=torch.long, device=device)
        combined = torch.cat((keep, all_idx))#把赛选出来的索引和原索引拼接起来，算出重复的（这些就是nms赛选出来的结果）
        uniques, counts = combined.unique(return_counts=True)#从大到小排序（优先次数然后）
        non_keep = uniques[counts == 1]#只出现了一次的锚框标号，即被筛选掉的锚框
        all_id_sorted = torch.cat((keep, non_keep))
        class_id[non_keep] = -1#直接屏蔽掉不需要输出的锚框
        class_id = class_id[all_id_sorted]#再按照输出优先度排序，这个输出所有的锚框
        conf, predicted_bb = conf[all_id_sorted], predicted_bb[all_id_sorted]#对概率和真实框重新排序
        # pos_threshold是一个用于非背景预测的阈值
        below_min_idx = (conf < pos_threshold)#筛选掉非常小的
        class_id[below_min_idx] = -1#根据bool决定是否赋值，赋值为-1的都是废物
        conf[below_min_idx] = 1 - conf[below_min_idx]#让废物的废物可能非常大（废物都是概率非常小的）
        #对两个一维向量进行升维[anchors,1],在dim=1上拼接大小为[anchors,1+1+4](种类，准确率，四位坐标)，而且可能性从大到小，应该不会存在同一类一起挤在前面的情况
        pred_info = torch.cat((class_id.unsqueeze(1),
                               conf.unsqueeze(1),
                               predicted_bb), dim=1)
        out.append(pred_info)
    return torch.stack(out)#[tensor[]]本质上类似于升维，将tensor连接起来?

anchors = torch.tensor([[0.1, 0.08, 0.52, 0.92], [0.08, 0.2, 0.56, 0.95],
                      [0.15, 0.3, 0.62, 0.91], [0.55, 0.2, 0.9, 0.88]])
offset_preds = torch.tensor([0] * anchors.numel())
cls_probs = torch.tensor([[0] * 4,  # 背景的预测概率
                      [0.9, 0.8, 0.7, 0.1],  # 狗的预测概率
                      [0.1, 0.2, 0.3, 0.9]])  # 猫的预测概率

fig = d2l.plt.imshow(img)
show_bboxes(fig.axes, anchors * bbox_scale,
            ['dog=0.9', 'dog=0.8', 'dog=0.7', 'cat=0.9'])

output = multibox_detection(cls_probs.unsqueeze(dim=0),
                            offset_preds.unsqueeze(dim=0),
                            anchors.unsqueeze(dim=0),
                            nms_threshold=0.5)
print(output)

fig = d2l.plt.imshow(img)
for i in output[0].detach().numpy():
    if i[0] == -1:
        continue
    label = ('dog=', 'cat=')[int(i[0])] + str(i[1])
    show_bboxes(fig.axes, [torch.tensor(i[2:]) * bbox_scale], label)

plt.show()