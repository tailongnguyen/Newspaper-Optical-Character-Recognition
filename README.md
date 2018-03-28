# VN Newspaper Digitalization


Using deep learning to extract text from Vietnamese digital newspaper. This work is inspired from this [article](https://arxiv.org/pdf/1506.04395.pdf). 

## Code explaination
Training data is generated automatically when training from a prior text file contain text samples.
The model input is width-dynamic but have a fixed height of 32. The workflow is image -> paragraphs -> lines -> preprocessing -> recognition.


### Prerequisites


```
keras==2.0.5
tensorflow==1.2.0
numpy
pillow
opencv
warpctc (for training only)
```

### Usage

I have not refined the code for training becauase installing warpctc is so complicated, but you can use pretrained weights for testing purpose.

Run:

```
python main.py
```

and then feed the image in prompt phase.


### Demo
![alt text](https://github.com/tailongnguyen/Newspaper-Optical-Character-Recognition/tree/master/images/0.jpg) 

```
02-375235 1 
```
![alt text](https://github.com/tailongnguyen/Newspaper-Optical-Character-Recognition/tree/master/images/46.png) 

```
NGUYỄN NGỌC THOẠI 
```
![alt text](https://github.com/tailongnguyen/Newspaper-Optical-Character-Recognition/tree/master/images/43.png) 

```
Không chỉ dừng lại ở chuyện ăn cắp vặt thông thường, các đối tượng trộm chó sẽ bị xử lý hình sự, chịu sự trừng trị của pháp luật vì hành vi trộm cắp của mình. 
```
![alt text](https://github.com/tailongnguyen/Newspaper-Optical-Character-Recognition/tree/master/images/9.jpg) 

```
lO1  nwiuisww1 DU ỤCH 1 rwuszame saxw CânĨho M "h ua thu P Ơ lưng chừng trời / Bìi, inh. HÀNG KIỀU Nắng như mật tỏa trên những đám mây bồng bênh, trải trên những nương lúa vàng ươm, thu trên caoở Bát Xát chiếm trọn con tim của kẻ miên xa lạc bước tới đây. Ở lưng chừng trời, sắc thu đẩy mộng mị.. ể MỮ.T,,T xa lạ với phượt thủ trên những cung đường Tây Bắc. Tuyến này đi suốt chiêu dài huyện miên núi Bát Xát (tỉnh Lào Cai) qua những làng bản, những ngôi nhà trình tường độc đáo và cả những ngôi làng xa tít ở độ cao chừng 3.000 mét so với mực nước biển... Hiếm nơi nào được như Bát Xát, đâu đâu cũng có những cung đường, cảnh thiên nhiên hút hồn lũ khách. Hành trình một vòng từ Sa Pa lên Mường Hum rôiY Tý,lại đi theo đường ven biên vẽ lại thành phố Lào Cai; là cung đường càng đi càng mê mẩn, để rổi phải trở lại bao lần. Bát Xát nằm ở độ cao khoảng 1.500 mét so với mực nước biển, riêng Y Tý cao khoảng 2.000 mét, không khí trong lành và mát mẻ quanh năm. Mùa đông, nhiệt độ vùng này xuống dưới 5 độ C, có nhữngnăm xảy ra hiện tượngbăng tuyết. Y Tý quanh năm mây phù vào mỗi sáng sớm và hoàng hôn, cũng là nơi ngắm mây đẹp nhất vùng Tầy Bảc. Lảo Thẩn của Y Tý là ngọn núi anh em với Fansipan. Cao gẩn 3.000 mét so với mực nước biển với cung đường đẩy gian nan, nhưng đỉnh Lảo Thẩn vẫn hấp dẫn nhờ những bản làng ở tuốt trên mây dọcđường đi. Buổi sáng, đứng trên đỉnh núi là biển mây mênh mông. Cả bình minh và hoànghôn nhìn từ ngọn núinày luôn rực rỡ mây ngũ sắc. Bát Xát "ccề . EITRI EHH EYnYXFE, 9IR)Fi GỊUFH Ệc33FmG GỨSÚF GGtBj2 li peween phân địnhbiêngiớihainước. Con 9u khách với mùa vàng Tây Đắc Thứ bảy Y Tý, Chủ nhât Q Mường Hum" Z Bìi, inh. HOẰNG KIỀu HỆ,3ÚT: - người Mông, Dao, Hà Nhì... ở trùng điệp núi non và thơ mộng của tiết trời Tây Bắc vừa chớm thu lành lạnh. Chợ phiên thường hợp từ tinh mơ. Có người phải thức dậy khi vừa qua ngày mới để xuống chợ. Cũng có người đi chợ từ đêm hôm trước rôi trở vê nhà khi quá nửa đêm. Còn du khách, muốn trải nghiệm hai phiên chợ này thì đi từ ngày trước nữa. Tứclà, bạn phải có mặt ở thành phố Lào Cai từ thú sáu để bắt đầu hànhtrìnhvenbiên rồi ngượclên núi cao để đếnY Tý Trên đường đi, sẽ dừng chân ở các điểm, như: Cửa khẩu Lào Cai dài lênY Tý tới cầu Thiên Sinh, túc từ Cột mốc 103 lùi về Cột mốc87, qua những bản làng và những ngôi nhà trình tường "đông ấm, hè mát" độc đáo của người Hà Nhì. Phải di chuyển từ ngày thứ sáu để bạn có mặt tại trung tâm xã Y Tý trong đêm và tham gia chợ phiên vào sáng sớm thứ bảy. Trên bản đồ du lịch, Y Tý không được nhắc tới nhiêu nhưng lại đặc biệt đối với phượt thủ và khách nước ngoài bởi khung cảnh thiên í W,Ií/ứắí; trước máy ảnh nhưng không hể có chuyện "năm nghìn mới cho chụp hình Hoạt động trao đổi hànghóa, muabán diễn ra một cách ốTímtit/i"iTh,Ỉ, của người bản địa, khácxa nhữngphiên chợ khác ở Sa Pa, Bác Hà (Lào Cai) hay Đông Văn, Mèo Vạc (Hà Giang). Đến trưa, chợ vãn. Du khách lại hành lý úIớKíT,KS: cánh đồng lớn gản liên với câu nói "Nhất Thanh, Nhì Lò, Tam Than, Tứ Tắc tứcbõn cánh đônglớn rộngtới cả trăm cây số vuông. Mường Hum không được rộng lớn như vậy nhung hơn hản về độ cao và hoành tráng so với Tú Mường. Chợ Mường Hum tấp nập và là chợ phiên lớn nhất Lào Cai. Khi sương mù còn dày đặc, đã nghe tiếng người lao xao ở chợ và kéo dài đến hai giờ chiểu. nẺỆX" Sắc màu chợ phiên Tây Bắc. Cũng nhu Y Tý, Mường Hum là phiên chợ chẳng mua hay bán thú gì. Kết thúc phiên còn giữ nguyên nét truyển thống nhungtấp chợ, bạn có thể về lại Sa Pa để khám phá thị nập người muakẻbán. Tất nhiên,ở đó,cũng trấn dulịchtrăm năm tuổi đẩy cổ kínhhoặc có những người tới chợ để gặp gỡ bạnbè, để xuôi Ô Quy Hồ sang Yên Bái để khám phá uống rượu ngô,ănbát thảng cố... rổi về chú đồng lúa Mù Căng Chải. o 
```


## Authors

[tailongnguyen](https://github.com/tailongnguyen)

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Hat tip to anyone who's code was used
* Inspiration
* etc
