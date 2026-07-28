# Nội dung điền cổng nộp Project II

## Thông tin cố định

- **Học phần:** IT3930E – Project II
- **Giảng viên hướng dẫn:** ThS. Nguyễn Duy Hiệp
- **Doanh nghiệp:** –
- **Quyền truy cập:** Không công khai (Không cho phép tìm kiếm và truy cập file báo cáo)

## Tên đề tài

Mô hình hóa phân phối nguồn và định tuyến bằng hỗn hợp Gaussian cho Flow Matching

Tên tiếng Anh dùng trên báo cáo:

> Gaussian Mixture Source Modeling and Routing for Flow Matching

## Mô tả đề tài

Đề tài nghiên cứu cách xây dựng phân phối nguồn học được cho Flow Matching
thay cho nguồn Gaussian chuẩn. Trước hết, một mô hình hỗn hợp Gaussian với ma
trận hiệp phương sai đường chéo được khớp trên latent VAE của dữ liệu ảnh.
Tiếp theo, một router tích chập được huấn luyện để xấp xỉ xác suất hậu nghiệm
của hỗn hợp và lựa chọn các thành phần dùng để sinh latent nguồn. Trên nền mô
hình DiT, đề tài so sánh hai cách tạo nguồn từ các thành phần được chọn (trung
bình có trọng số và lấy mẫu một thành phần), hai miền dữ liệu huấn luyện router
(các điểm đầu mút và các điểm nội suy), cùng các biến thể về capacity,
regularization và khởi tạo GMM.

Phần triển khai tái sử dụng backbone DiT, pipeline dữ liệu và chức năng Flow
Matching trong repo Shortcut Models. Đề tài không huấn luyện mục tiêu
bootstrapping của Shortcut Models: với chế độ `gmm-tide`, đầu vào biểu diễn
bước nhảy được giữ cố định qua `ignore_dt=True`, còn mục tiêu chính là hồi quy
trường vận tốc từ latent nguồn đã định tuyến đến latent dữ liệu.

Phần thực nghiệm đánh giá chất lượng ảnh bằng FID tại các checkpoint khớp nhau.
Mỗi checkpoint được đo lại trên nhiều seed sinh ảnh và toàn bộ thiết lập được
lặp với nhiều lần khớp GMM để tách nhiễu đánh giá khỏi độ bất định của phân
phối nguồn. Mục tiêu là xác định yếu tố nào thực sự ảnh hưởng đến chất lượng
sinh ảnh và xây dựng quy trình đánh giá có thể tái lập, thay vì chọn mô hình
chỉ từ một kết quả FID đơn lẻ.

## Project description in English

This project investigates learned source-distribution modeling for Flow
Matching as an alternative to the standard Gaussian source. First, a diagonal
Gaussian mixture model is fitted to the VAE latents of the image data. A
convolutional router is then trained to approximate the mixture posterior and
select the components used to generate source latents. Using a DiT backbone,
the project compares two source-construction methods (weighted averaging and
sampling a single selected component), two router training domains (endpoint
mixtures and interpolated points), and several variations in router capacity,
regularization, and GMM initialization.

The implementation reuses the DiT backbone, data pipeline, and Flow Matching
functionality of the Shortcut Models repository. It does not train the shortcut
bootstrapping objective. In the gmm-tide mode, the step-size input is held
constant through ignore_dt=True, while the main objective is velocity-field
regression from a routed source latent to a data latent.

The experiments evaluate image quality using FID at matched checkpoints. Each
checkpoint is evaluated with multiple image-generation seeds, and the complete
setup is repeated across multiple GMM fits to distinguish evaluation noise
from uncertainty in the learned source distribution. The objective is to
identify which factors materially affect image quality and to establish a
reproducible evaluation procedure, rather than selecting a model from a single
favorable FID result.

## Từ khóa

Flow Matching; Gaussian Mixture Model; VAE latent; neural routing; Diffusion
Transformer; Fréchet Inception Distance

## Tóm tắt báo cáo (dán vào ô trên cổng nộp)

**(1) Nội dung công việc được giao.** Đề tài tập trung nghiên cứu mô hình hóa
phân phối nguồn cho Flow Matching trong không gian latent của VAE. Các công
việc chính gồm: tìm hiểu các khái niệm VAE latent, Gaussian Mixture Model
(GMM), Flow Matching, Diffusion Transformer (DiT), neural router và FID; khớp
GMM đường chéo trên latent dữ liệu; huấn luyện router tích chập để xấp xỉ hậu
nghiệm GMM; xây dựng các phương án tạo latent nguồn từ những thành phần được
chọn; và thiết kế các thí nghiệm đối chứng để đánh giá ảnh hưởng của miền huấn
luyện router, cách lấy mẫu nguồn, capacity, regularization và khởi tạo GMM.
Phần triển khai sử dụng chức năng Flow Matching và backbone DiT của repo
Shortcut Models, không huấn luyện mục tiêu shortcut bootstrapping.

**(2) Kết quả thực hiện.** Đề tài đã xây dựng được pipeline gồm khớp GMM, huấn
luyện router, huấn luyện Flow Matching, ghi log chẩn đoán, sinh ảnh và đánh giá
FID. Thí nghiệm chính là factorial 2 × 2 tại checkpoint 400k, so sánh
hai cách tạo nguồn (weighted và sample-topk) với hai miền huấn luyện router
(mix và bridge). Mỗi cấu hình được lặp trên ba GMM seed và mỗi checkpoint
được đánh giá bằng năm seed sinh ảnh. Trong factorial, cấu hình sample-topk
kết hợp mix đạt FID trung bình tốt nhất là 7.046. Tuy nhiên, kết quả này chưa
vượt mốc tham chiếu lịch sử 6.969, vốn chỉ là một phép đo đơn tại checkpoint
không khớp. Phân tích cũng cho thấy biến thiên do khớp lại GMM lớn hơn nhiễu
sinh ảnh trong cùng checkpoint; các chỉ số router, độ cong của flow và mức cân
bằng component có giá trị chẩn đoán nhưng chưa dự đoán ổn định thứ hạng FID.

**(3) Tự đánh giá kết quả thực hiện.** Ưu điểm của phần thực hiện là pipeline
thí nghiệm và thu thập bằng chứng đã được chuẩn hóa; các so sánh chính sử dụng
checkpoint khớp, repeated evaluation và nhiều GMM seed; lỗi hạ tầng được tách
khỏi kết quả mô hình; đồng thời báo cáo không lựa chọn kết luận chỉ từ một giá
trị FID tốt nhất. Nhược điểm là factorial mới có ba lần khớp GMM, chưa có đủ
seed huấn luyện end-to-end độc lập và baseline lịch sử chưa được đánh giá lại
theo cùng protocol. Vì vậy, kết quả hiện tại xác định được xu hướng, nguồn bất
định và hướng cải tiến tiếp theo, nhưng chưa đủ để khẳng định phương pháp mới
vượt baseline một cách ổn định.

## Report summary in English (copy into the submission portal)

**(1) Assigned work.** This project investigates source-distribution modeling
for Flow Matching in the VAE latent space. The assigned work included studying
VAE latents, Gaussian mixture models (GMMs), Flow Matching, Diffusion
Transformers (DiTs), neural routing, and FID; fitting a diagonal GMM to data
latents; training a convolutional router to approximate the GMM posterior;
constructing source latents from selected mixture components; and designing
controlled experiments to assess the effects of the router training domain,
source-sampling method, model capacity, regularization, and GMM
initialization. The implementation uses the Flow Matching functionality and
DiT backbone of the Shortcut Models repository, without training the shortcut
bootstrapping objective.

**(2) Results.** The completed pipeline covers GMM fitting, router training,
Flow Matching training, diagnostic logging, image generation, and FID
evaluation. The main experiment is a 2 × 2 factorial study at the
400k checkpoint, comparing two source constructors (weighted and sample-topk)
and two router training domains (mix and bridge). Each configuration was
repeated across three GMM seeds, and every checkpoint was evaluated with five
image-generation seeds. Within this factorial study, sample-topk combined with
mix achieved the best mean FID of 7.046. However, it did not outperform the
historical reference of 6.969, which was obtained from a single evaluation at
an unmatched checkpoint. The analysis also shows that variation caused by
refitting the GMM is larger than image-generation noise within a fixed
checkpoint. Router metrics, flow curvature, and component balance are useful
diagnostics, but they do not reliably predict the FID ranking.

**(3) Self-assessment.** The main strengths are a standardized experimental and
evidence-collection pipeline, matched-checkpoint comparisons, repeated
evaluation across multiple GMM seeds, explicit separation of infrastructure
failures from model results, and conclusions that do not rely on a single best
FID value. The limitations are that the factorial study contains only three
GMM fits, does not include enough independent end-to-end training seeds, and
does not reevaluate the historical baseline under the same protocol. The
current results therefore identify empirical trends, sources of uncertainty,
and directions for further improvement, but they are not sufficient to claim
that the proposed method consistently outperforms the baseline.

## Ghi chú xác minh học vị người hướng dẫn

Trang hồ sơ chính thức của SOICT ghi **Thạc sĩ (Đại học Bách khoa Hà Nội,
2010)** và trang danh sách cán bộ tiếng Anh ghi **M.S. Nguyen Duy Hiep**. Vì
vậy dùng **ThS. Nguyễn Duy Hiệp** trong biểu mẫu tiếng Việt và
**M.Sc. Nguyễn Duy Hiệp** trên bìa tiếng Anh; không dùng **Ph.D.**

- https://soict.hust.edu.vn/ths-nguyen-duy-hiep.html
- https://soict.hust.edu.vn/en/officer/page/4
- https://hust.edu.vn/uploads/sys/news/2024_05/dhbkhn_de-an-tuyen-sinh-2024_1.pdf
