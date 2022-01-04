Tham Khảo [paper](https://arxiv.org/abs/1801.02143)

Tham Khảo [posts](https://towardsdatascience.com/build-a-handwritten-text-recognition-system-using-tensorflow-2326a3487cd5)

Kiến trúc mạng chính là sự kết hợp: CNN + RNN + Transcription

Trancription: [Tham Khảo CTC](https://distill.pub/2017/ctc/)

Tập dữ liệu sử dụng trong bài này [IAM](https://fki.tic.heia-fr.ch/databases/iam-handwriting-database) |
[Data VN](https://miai.vn/download.phpurl=https://www.mediafire.com/file/5a1k0rsmkrhm4pm/HandWriting.zip/file)

Tìm hiểu thuật toán CTC (Connectionist Temporal Classification) từ bài viết [Sequence Modeling
With CTC](https://distill.pub/2017/ctc/):
- Thuật toán CTC nó làm 2 công việc chính đó là tính toán hàm chi phí và dự đoán(suy luận).
- Thật ra, thuật toán CTC nó không có căn chỉnh - liên kết, nó không yêu cầu sự căn chỉnh giữa đầu vào và đầu ra. Mặc dù là như vậy nhưng để có xác suất của đầu ra cho trước một đầu vào thì nó làm việc bằng cách tính tổng hợp các xác suất của tất cả liên kể có thể có của cả 2. Chúng ta cần hiểu những căn chỉnh này là gì để hiểu cách tính hàm tổn thất cuối cùng.
- VD: Cho 1 đầu vào X có độ dài là 6 và Y = [c,a,t] Một cách liên kết giữa X và Y là ta sẽ gán ký tự đầu ra cho mỗi đầu vào và lặp lại
![image](https://user-images.githubusercontent.com/72034584/148024601-f63fcc28-6e47-49e8-a361-e61975000410.png)
- Tuy nhiên bạn sẽ thấy có 1 số vấn đề:
  - Nó sẽ không có ý nghĩa gì nếu buộc mọi bước đầu vào phải phù hợp với một số đầu ra. VD: Nếu đầu vào của bạn 1 background không chứa từ nào mà không có  đầu ra tương ứng   
  - Chúng tôi không có cách nào để tạo ra kết quả có nhiều ký tự trong một hàng. Xét sự thẳng hàng [h, h, e, l, l, l, o]. Việc thu gọn các lần lặp lại sẽ tạo ra "helo" thay vì "hello"
- Từ vấn đề này thì CTC có thể giải quyết triệt để đó là, CTC sẽ chèn thêm vào ký tự khoảng trắng.
![image](https://user-images.githubusercontent.com/72034584/148026137-f6f40b02-952b-4001-98c9-79ec258031a5.png)
- Chúng ta sẽ thêm vào khoảng trắng vào nếu các ký tự giống nhau và liên kề. Với quy tắc này, chúng ta có thể phân biệt giữa các liên kết thu gọn thành "hello" và những liên kết thu gọn thành "helo".
- Một số trường hợp thêm khoảng trắng:
![image](https://user-images.githubusercontent.com/72034584/148026751-2060fd41-7b1f-4854-91c4-3fc468274577.png)
- Một số liên kết CTC cần chú ý:
  - Sự liên kết giữa X và Y mang tính đơn điệu. Nếu chúng ta chuyển sang đầu vào tiếp theo, chúng ta có thể giữ nguyên đầu ra tương ứng hoặc chuyển sang đầu tiếp theo.
  - Độ dài của Y không được lớn X
  - Tính chất thứ hai là sự liên kết của X thành Y là nhiều-một. Một hoặc nhiều phần tử đầu vào có thể liên kết với một phần tử đầu ra duy nhất nhưng không thể ngược lại.
- Thực chất, bản thân CTC cũng ko biết chính xác cách Alignment giữa X và Y. Nó làm việc theo kiểu Work Arround, tức là nếu đưa cho nó X thì nó sẽ trả lại cho ta tất cả các khả năng của Y, kèm theo xác suất chính xác của mỗi khả năng đó.
- **Loss function**:

![image](https://user-images.githubusercontent.com/72034584/148027996-85cca589-6c04-4c50-a0c9-a11fcb6e6299.png)

- Ở hàng đầu tiên chúng ta có đó input là feature maps -> Đầu vào của mạng RNN ở đây thì 10 time-step -> Mạng sẽ tính toán xác suất cho mỗi input {h,e,l,o,-} -> Tính toán xác suất có thể có, màu càng đậm thì xác suất càng cao -> cách ly so với tập hợp các căn chỉnh hợp lệ (output)

![image](https://user-images.githubusercontent.com/72034584/148028450-99e68f89-372b-4346-b169-1af19b9f1f2c.png)

-  Nó là tổng tất cả các Scores của tất cả các khả năng Alignments của output.
-  VD: Nó sẽ đi tính tổng của từng ký tự rồi nó tổng cả ký tự lại:
   -  Các khả năng Alignment của ký tự h là: hhh, h–, h-, hh-, -hh, –h –> Score của *h = 0.4x0.3x0.4 + 0.4x0.7x0.6 + 0.4x0.7 + 0.4x0.3x0.6 + 0.1x0.3x0.4 + 0.1x0.7x0.4 = 0.608
   -  Các khả năng Alignment của ký tự e là: eee, e–, e-, ee-, -ee, –e –> Score của *a = 0.4x0.3x0.4 + 0.4x0.7x0.6 + 0.4x0.7 + 0.4x0.3x0.6 + 0.1x0.3x0.4 + 0.1x0.7x0.4 = 0.2
   -  Các khả năng Alignment của ký tự l là: lll, l–, l-, ll-, -ll, –l –> Score của *l = 0.4x0.3x0.4 + 0.4x0.7x0.6 + 0.4x0.7 + 0.4x0.3x0.6 + 0.1x0.3x0.4 + 0.1x0.7x0.4 = 0.6
   -  Các khả năng Alignment của ký tự o là: ooo, oo–, o-, oo-, -oo, –o –> Score của *o = 0.4x0.3x0.4 + 0.4x0.7x0.6 + 0.4x0.7 + 0.4x0.3x0.6 + 0.1x0.3x0.4 + 0.1x0.7x0.4 = 0.608
   -> Tổng Loss = 0.216 + 0.2 + 0.6 + 0.608
   Sau khi có được hàm loss, chúng ta có thể tính toán gradient như thông thường. Tham số sẽ được điều chỉnh để minimize hàm negative log-likelihood.
- **Decoding text**
- Tìm đường đi tối ưu nhất từ Score Matrix bằng cách chọn các ký tự có Score cao nhất tại mỗi TimeStep. Xóa bỏ các ký tự trống, ký tự trùng lặp.
- ![Web capture_4-1-2022_153441_distill pub](https://user-images.githubusercontent.com/72034584/148031339-5be32e42-e196-494c-8bf2-0919f7d0ca08.jpeg)

=> Ý tưởng cơ bản thuật toán hoạt động như vậy, tuy nhiên còn nhiều vấn đề còn nghiên cứu thêm nữa. Khuyết khích đọc bài viết mình có gắn link ở trên
